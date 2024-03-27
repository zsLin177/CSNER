import argparse
from supar.utils.logging import init_logger, logger
from supar.utils import Config
import os
import torch
from my_parser import Parser

def parse(parser):
    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--unlabel_file', default=None, type=str)
    parser.add_argument('--init_model_path', default=None, type=str)
    parser.add_argument('--roberta_path', default="chinese-roberta-wwm-ext-large", type=str)
    parser.add_argument('--device', default="-1", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n_workers', default=6, type=int)
    parser.add_argument('--pad_index', default=0, type=int)
    parser.add_argument('--bos_index', default=101, type=int)
    parser.add_argument('--bert_req_grad', action='store_true')
    parser.add_argument('--icsr', action='store_true')
    
    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.manual_seed(args.seed)
    # for torch 2.0
    torch.set_float32_matmul_precision('high')

    if args.mode == "predict":
        init_logger(logger, args.output_file+'.log')
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.predict()
    elif args.mode == "self-train":
        init_logger(logger, os.path.join(args.path, f'{args.mode}.log'))
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.self_training()
    elif args.mode == "self-trans-train":
        init_logger(logger, os.path.join(args.path, f'{args.mode}.log'))
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.self_trans_training()
    elif args.mode == "tri-train":
        init_logger(logger, os.path.join(args.path, f'{args.mode}.log'))
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.tri_training()
    elif args.mode == "evaluate":
        init_logger(logger, args.input_file+'-evaluate.log')
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.evaluate()
    elif args.mode == "train":
        init_logger(logger, os.path.join(args.path, f'{args.mode}.log'))
        logger.info('\n' + str(args))
        parser = Parser(args)
        parser.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', dest='mode')

    # self-training
    subparser = subparsers.add_parser('self-train')
    subparser.add_argument('--select_file_dir', default=None, type=str, required=True, help="the directory of the selected data")
    subparser.add_argument('--src_train_file', default=None, type=str, required=True, help="the source train file")
    subparser.add_argument('--path', default=None, type=str, required=True, help="the dir to save the model")
    subparser.add_argument('--dev_file', default=None, type=str, required=True, help="the file to dev")
    subparser.add_argument('--test_file', default=None, type=str, required=True, help="the file to test")
    subparser.add_argument('--iter', default=5, type=int, help="the number of iteration for self training")
    subparser.add_argument('--p_hold', default=0.9, type=float, help="the min probability of label squence for holding the data")
    subparser.add_argument('--p_drop', default=0.5, type=float, help="the probability of dropping the data that is all O")
    subparser.add_argument('--update_steps', default=1, type=int)
    subparser.add_argument('--epochs', default=20, type=int)
    subparser.add_argument('--lr', default=2e-4, type=float)
    subparser.add_argument('--lr_rate', default=1, type=float)
    subparser.add_argument('--warmup', default=0.1, type=float)
    subparser.add_argument('--patience', default=5, type=int)
    subparser.add_argument('--use_kl', action='store_true')
    subparser.add_argument('--kl_weight', default=0.1, type=float)
    subparser.add_argument('--dynamic_p_hold', action='store_true')
    subparser.add_argument('--onasr', action='store_true')
    subparser.add_argument('--onlycer', action='store_true')

    # self-transform-training
    subparser = subparsers.add_parser('self-trans-train')
    subparser.add_argument('--select_file_dir', default=None, type=str, required=True, help="the directory of the selected data")
    subparser.add_argument('--src_train_file', default=None, type=str, required=True, help="the source train file")
    subparser.add_argument('--prd_transformed_data', default=None, type=str, required=True, help="the predicted transformed data by the baseline model, and the label is mapped back to the original text")
    subparser.add_argument('--trans_iter', default=1, type=int, help="the number of iteration for use the predicted transformed data")
    subparser.add_argument('--path', default=None, type=str, required=True, help="the dir to save the model")
    subparser.add_argument('--dev_file', default=None, type=str, required=True, help="the file to dev")
    subparser.add_argument('--test_file', default=None, type=str, required=True, help="the file to test")
    subparser.add_argument('--iter', default=10, type=int, help="the number of iteration for self training")
    subparser.add_argument('--p_hold', default=0.9, type=float, help="the min probability of label squence for holding the data")
    subparser.add_argument('--p_drop', default=0.0, type=float, help="the probability of dropping the data that is all O")
    subparser.add_argument('--update_steps', default=1, type=int)
    subparser.add_argument('--epochs', default=20, type=int)
    subparser.add_argument('--lr', default=2e-4, type=float)
    subparser.add_argument('--lr_rate', default=1, type=float)
    subparser.add_argument('--warmup', default=0.1, type=float)
    subparser.add_argument('--patience', default=5, type=int)
    subparser.add_argument('--use_kl', action='store_true')
    subparser.add_argument('--kl_weight', default=0.1, type=float)

    # tri-training
    subparser = subparsers.add_parser('tri-train')
    subparser.add_argument('--select_file_dir', type=str, required=True, help="the directory of the selected data")
    subparser.add_argument('--src_train_file', type=str, required=True, help="the source train file")
    subparser.add_argument('--path', type=str, required=True, help="the dir to save the model")
    subparser.add_argument('--init_aff_model_path', type=str, required=True)
    subparser.add_argument('--dev_file', default=None, type=str, required=True, help="the file to dev")
    subparser.add_argument('--test_file', default=None, type=str, required=True, help="the file to test")
    subparser.add_argument('--iter', default=10, type=int, help="the number of iteration for self training")
    subparser.add_argument('--update_steps', default=1, type=int)
    subparser.add_argument('--epochs', default=20, type=int)
    subparser.add_argument('--lr', default=2e-4, type=float)
    subparser.add_argument('--lr_rate', default=1, type=float)
    subparser.add_argument('--warmup', default=0.1, type=float)
    subparser.add_argument('--patience', default=3, type=int)


    # train
    subparser = subparsers.add_parser('train')
    subparser.add_argument('--train_file', default=None, type=str, required=True)
    subparser.add_argument('--dev_file', default=None, type=str, required=True)
    subparser.add_argument('--test_file', default=None, type=str, required=True)
    subparser.add_argument('--path', default=None, type=str, required=True)
    subparser.add_argument('--update_steps', default=1, type=int)
    subparser.add_argument('--epochs', default=25, type=int)
    subparser.add_argument('--lr', default=5e-4, type=float)
    subparser.add_argument('--lr_rate', default=1, type=float)
    subparser.add_argument('--warmup', default=0.1, type=float)
    subparser.add_argument('--patience', default=5, type=int)
    subparser.add_argument('--use_kl', action='store_true')
    subparser.add_argument('--kl_weight', default=0.1, type=float)
    subparser.add_argument('--onasr', action='store_true')
    subparser.add_argument('--onlycer', action='store_true')

    # predict
    subparser = subparsers.add_parser('predict')
    subparser.add_argument('--input_file', default=None, type=str)
    subparser.add_argument('--output_file', default=None, type=str)
    subparser.add_argument('--path', default=None, type=str, required=True, help="the path to the saved model")

    # evaluate
    subparser = subparsers.add_parser('evaluate')
    subparser.add_argument('--input_file', default=None, type=str)
    subparser.add_argument('--path', default=None, type=str, required=True, help="the path to the saved model")
    subparser.add_argument('--onasr', action='store_true')
    subparser.add_argument('--onlycer', action='store_true')

    parse(parser)
