"""Example: python cli.py --x 0 1 0 --y 1 .5 0 .5"""
import argparse,json
from monitor import evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Outward-arithmetic stopping for the nominal ideal four-analyzer quadratic.')
    parser.add_argument('--x',nargs=3,type=float,required=True,metavar=('A','W0','W1'))
    parser.add_argument('--y',nargs=4,type=float,required=True,metavar=('Y0','Y45','Y90','Y135'))
    parser.add_argument('--bits',type=int,default=128)
    args=parser.parse_args();print(json.dumps(evaluate(args.x,args.y,args.bits),indent=2,allow_nan=False))
