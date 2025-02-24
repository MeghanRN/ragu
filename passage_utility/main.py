from importlib import util
from scipy.stats import mode
import torch
from load_data import load_ragqa
import sys
sys.path.append('../utils')
from utils import save_file_jsonl
import os
import numpy as np
import argparse
from reward_learner.vallina_bert import VanillaBert
from models.bert_ranker import BertRanker
from tqdm import trange
import warnings
import swag.utils as utils
from evaluator.evaluation import evaluateReward
from torch.utils.data import DataLoader
from dataset_collection import PosNegDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_shards', type=int, default=0)          

    parser.add_argument('--lr_init', type=float,
                        default=5e-5, help='initial learning rate')
    parser.add_argument('--ilr', type=float, default=1e-4,
                        metavar='N', help='learning rate for interaction')
    parser.add_argument('--wd', type=float, default=1e-2,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=3,
                        metavar='N', help='SWA start epoch number')
                  
    parser.add_argument('--proportion', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to train')
    parser.add_argument('--proportion_dev', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to from dev set')                        

    parser.add_argument('--batch_size', type=int, default=16,
                        metavar='N', help='input batch size')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased',
                        metavar='N', help='the name of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        metavar='N', help='dataset')
    parser.add_argument('--test_mode', type=str, default='test',
                        metavar='N', help='dataset')

    parser.add_argument('--sample_nums', type=int, default=20, metavar='N')
    parser.add_argument('--model_name', type=str, default='vanilla_bert')
    parser.add_argument('--margin', type=float, default=0.1, metavar='N')  
    parser.add_argument('--stop_epochs', type=int, default=3, metavar='N')

    parser.add_argument('--interactive', type=bool, default=False, metavar='N')
    parser.add_argument('--do_train', type=bool, default=None, metavar='N')
    parser.add_argument('--do_test', type=bool, default=None, metavar='N')
    parser.add_argument('--pool_sample', type=bool, default=False, metavar='N')    

    parser.add_argument('--format_ques', type=bool, default=False, metavar='N')    
    parser.add_argument('--add_title', type=bool, default=False, metavar='N')
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--reference_rank', type=str, default='retriever',
                        help="criteria to use as reference utility (other possible values: rl-nli, rl, nli, acc-nli, acc_LM-nli, acc, acc-ties).")            
    parser.add_argument('--output_pred_utilities', type=bool, default=False, metavar='N')
    parser.add_argument('--combine_loss', type=str, default=None,
                        help="whether to combine ranking loss with BE/MSE(values: be, mse).")      
    parser.add_argument('--weight_rank', type=float, default=1,
                        help="weight of ranking loss.")   
    parser.add_argument('--weight_aux', type=float, default=0,
                        help="weight of combined auxiliary loss.")   
    parser.add_argument(
        "--model_select", type=str, default="combined",
        choices=['combined', 'error', 'rank'],
        help="Model selection criteria for improvement evaluation.")                                                

    args = parser.parse_args()

    print(args)
    directory_name = os.path.join(args.save_dir,'logs')
    log_mode = 'TRAIN' if args.do_train else f'EVAL:{args.test_mode}'
    log_file_name = os.path.join(directory_name, f'{args.model_name}-{log_mode}.log')
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    log_file = open(log_file_name, 'w')
    log_file.write(f'{args}')
    print(f'Logging run config into: {log_file_name}\n')

    warnings.filterwarnings('ignore')
    if args.interactive:
        print('Not implemented')
        exit(0)
        # load data
        qa_list, ref_values = load_ragqa(
            args.input_file.replace('SPLIT', args.mode), args.top_n, args.reference_rank, interactive=True)
            
    # data
    if args.do_train:
        data_pair = load_ragqa(
            args.input_file.replace('SPLIT', args.mode), args.top_n, args.reference_rank, interactive=False, 
                    shards=args.num_shards, single_net=(args.weight_rank==0), add_title=args.add_title)
        dev_pair = load_ragqa(
            args.input_file.replace('SPLIT', 'dev'), args.top_n, args.reference_rank, interactive=False,
                    single_net=(args.weight_rank==0), add_title=args.add_title)
        
        print('training data size', len(data_pair))
        print('dev data size', len(dev_pair))
        data_loader = DataLoader(PosNegDataset(
            data_pair[:int(len(data_pair)*args.proportion)], args.pretrained_model), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(PosNegDataset(
            dev_pair[:int(len(dev_pair)*args.proportion_dev)], args.pretrained_model), batch_size=args.batch_size, shuffle=False)

    # select data(batch or single) according to query strategy, initial data 0.1%?
    # querier
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    base_model = BertRanker(args.pretrained_model)
    base_model.to(device)

    # train
    if args.do_train:
        print('start training')
        if 'vanilla_bert' in args.model_name.lower():
            model = VanillaBert(base_model, lr=args.lr_init, ilr=args.ilr, epochs=args.epochs,
                             pretrained_model=args.pretrained_model, device=device, weight_decay=args.wd, 
                             margin=args.margin, combine_loss=args.combine_loss, weight_rank=args.weight_rank, 
                             weight_aux=args.weight_aux, model_select=args.model_select)
            model.to(device)
        else:
            print('Unimplemented')
            exit(0)
            
        if args.checkpoint:
            checkpoint = torch.load(os.path.join(
                args.save_dir, args.checkpoint+'.pt'))
        else:
            checkpoint = None
        model.train(data_loader, valid_loader=valid_loader, save_dir=args.save_dir,
                stop_epochs=args.stop_epochs, save_name=args.model_name, checkpoint =checkpoint,
                log_file=log_file)


    # Predict taking a sample
    if args.pool_sample:
        # test mode and interactive use the same loaded format
        test_qa_list, test_ref_values = load_ragqa(
            args.input_file.replace('SPLIT', args.test_mode), args.top_n, args.reference_rank, interactive=True)

        if 'vanilla_bert' in args.model_name.lower():
            model = VanillaBert(base_model, lr=args.lr_init, ilr = args.ilr, epochs=args.epochs,
                             pretrained_model=args.pretrained_model, device=device, weight_decay=args.wd, 
                             margin=args.margin, combine_loss=args.combine_loss, weight_rank=args.weight_rank, weight_aux=args.weight_aux)
            checkpoint = torch.load(os.path.join(
                args.save_dir, args.model_name+'best-val-acc-model'+'.pt'))
            model.to(device)
        else:
            print('Unimplemented')
            exit(0)
        
        model.load_state_dict(checkpoint['state_dict'])

        print('finish loading model!')
        for question_id in trange(len(test_qa_list)):
            # get the current best estimate
            entry = test_qa_list[question_id]
            if args.add_title:          
                test_data = [ctx['title'] + '\n' + ctx['text'] for ctx in entry['ctxs'][:args.top_n]]
            else:
                test_data = [ctx['text'] for ctx in entry['ctxs'][:args.top_n]]
            question = entry['question']
            if args.format_ques:
                question = question[0].upper() + ques[1:] + '?'
            questions = [question]*len(test_data)
            pooled_mean, pooled_var = model.predict(test_data, questions, args.sample_nums, eval=False) ## to predict with MCD

            for i, ctx in enumerate(entry['ctxs'][:args.top_n]):
                ctx['mean'] = pooled_mean[i]
                ctx['var'] = pooled_var[i]

        if args.output_pred_utilities:
            closs='_' + "{:.2f}".format(args.weight_aux) + args.combine_loss if args.combine_loss else ''
            tit = '' if not args.add_title else '_wTITLE'
            result_fp = args.input_file.replace('.jsonl',f'-distrib_{args.weight_rank}{args.reference_rank}{closs}_{args.model_select}_pred{tit}.jsonl')
            save_file_jsonl(test_qa_list, result_fp)
            print('Files saved to, ', result_fp)


    if args.do_test:
        # test mode and interactive use the same loaded format
        test_qa_list, test_ref_values = load_ragqa(
            args.input_file.replace('SPLIT', args.test_mode), args.top_n, args.reference_rank, interactive=True)

        print('start testing...')

        checkpoint = torch.load(os.path.join(
                args.save_dir, args.model_name+'best-val-acc-model'+'.pt'))
            
        if 'vanilla_bert' in args.model_name.lower():
            print('vallina bert!!')
            model = VanillaBert(base_model, lr=args.lr_init, ilr = args.ilr, epochs=args.epochs,
                             device=device, pretrained_model=args.pretrained_model, weight_decay=args.wd, 
                             margin=args.margin, combine_loss=args.combine_loss, weight_rank=args.weight_rank, weight_aux=args.weight_aux)
        else:
            print('Unimplemented')
            exit(0)

        model.load_state_dict(checkpoint['state_dict'])
        print('finish loading model!')

        res, acc = 0, 0
        whole_answers, question, cumcnt = [], [], []
        prev = 0
        for question_id in trange(len(test_qa_list)):
            entry = test_qa_list[question_id]
            if args.add_title:
                pooled = [ctx["title"] + "\n" + ctx["text"] for ctx in entry["ctxs"][:args.top_n]]
            else:
                pooled = [ctx["text"] for ctx in entry["ctxs"][:args.top_n]]
            if args.format_ques:
                ques = entry["question"]
                ques = ques[0].upper() + ques[1:] + '?'
                question.extend([ques]*len(pooled))
            else:
                question.extend([entry["question"]]*len(pooled))
            whole_answers.extend(pooled)
            prev += len(pooled)
            cumcnt.append(prev)
        print('total nums', len(whole_answers))
        assert len(question) == len(whole_answers)
        utilities = model.get_utilities(
            test_data=whole_answers, question=question, sample_nums=args.sample_nums)
        print('Compute ranking eval metrics!')
        for i in trange(len(cumcnt)):
            if i == 0:
                single_utility = utilities[:cumcnt[i]]
            else:
                single_utility = utilities[cumcnt[i-1]:cumcnt[i]]
            gold_values = test_ref_values[i]
            assert len(single_utility) == len(gold_values)
            acc += (np.argmax(single_utility) == np.argmax(gold_values))
            metric_dict = evaluateReward(single_utility, gold_values)
            res += metric_dict['ndcg_at_all'] #'ndcg_at_all' 'ndcg_at_5%'

        if args.output_pred_utilities:
            i = 0
            for i, item in enumerate(test_qa_list):
                if i == 0:
                    single_utility = utilities[:cumcnt[i]]
                else:
                    single_utility = utilities[cumcnt[i-1]:cumcnt[i]]
                assert len(single_utility) == len(gold_values)
                for j in range(args.top_n):
                    item["ctxs"][j][args.reference_rank + "_pred"] = single_utility[j]

            closs='_' + "{:.2f}".format(args.weight_aux) + args.combine_loss if args.combine_loss else ''
            tit = '' if not args.add_title else '_wTITLE'
            result_fp = args.input_file.replace('.jsonl',f'-point_{args.weight_rank}{args.reference_rank}{closs}_{args.model_select}_pred{tit}.jsonl')
            save_file_jsonl(test_qa_list, result_fp)
            print('Files saved to, ', result_fp)

        print('res', res/len(test_qa_list))
        print('acc', acc/len(test_qa_list))

    log_file.close()
