import argparse
from chatgpt.gptner import process, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', dest='mode')

    subparser = subparsers.add_parser('predict', help='use openai api to predict')
    subparser.add_argument("--input_jsonl_file", type=str)
    subparser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")
    subparser.add_argument("--shot", type=int, default=3)
    subparser.add_argument("--method", type=str, choices=["ST", "QA"]) # special token (ST), QA
    subparser.add_argument("--onasr", action="store_true")
    subparser.add_argument("--icsr", action="store_true")

    subparser = subparsers.add_parser('evaluate', help='evaluate the results')
    subparser.add_argument("--input_jsonl_file", type=str)
    subparser.add_argument("--method", type=str, choices=["ST", "QA"]) # special token (ST), QA
    subparser.add_argument("--onasr", action="store_true")
    subparser.add_argument("--icsr", action="store_true")

    args = parser.parse_args()
    if args.mode == "predict":
        process(args.input_jsonl_file, args.model, args.method, args.shot, args.onasr, args.icsr)

    elif args.mode == "evaluate":
        evaluate(args.input_jsonl_file, args.method, args.onasr, args.icsr)