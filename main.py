from spline_learner_poe4d_MM import *
import sys, getopt
import argparse
import os
import sys
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-MM", "--use_mm", help="use Michaelis Menten",type=int)
    parser.add_argument("-a","--aval",help = "a value",type=str)
    parser.add_argument("-b","--bval", help = "b value",type=float)
    parser.add_argument("-nb", "--num_bact", help="number OTUS", type=int)
    parser.add_argument("-mv", "--mvar", help="meas var", type=float)
    parser.add_argument("-pv", "--pvar", help="proc var", type=float)
    parser.add_argument("-poe", "--poe_var", help="poe var", type=float)
    parser.add_argument("-tm", "--time", help="total time", type=int)
    parser.add_argument("-no", "--nobs",
                        help="number of observations", type=int)
    parser.add_argument("-dt", "--delta_t", help="delta t", type=float)
    parser.add_argument("-gr", "--growth_rate", help="growth rate", type=float)
    parser.add_argument("-av", "--avar", help="a variance", type=float)
    parser.add_argument("-bv", "--bvar", help="b variance", type=float)
    parser.add_argument("-tv", "--theta_var",
                        help="theta variance", type=float)
    parser.add_argument("-gstepps", "--gibbs_steps",
                        help="number of steps", type=int)
    
    parser.add_argument("-o", "--outdir", help="out directory", type=str)
    parser.add_argument("-useF1", "--useF1", help="use f1 or not", type=bool)

    args = parser.parse_args()

    curr = os.getcwd()
    if args.outdir == None:
        print('Specifiy Outdir')
        sys.exit(0)
    elif not os.path.exists(curr  + '/' + args.outdir):
        os.mkdir(curr + '/' + args.outdir)
    # if args.use_mm and args.aval and args.bval and args.num_bact and args.mvar and args.pvar and args.theta_var and args.avar and args.bvar and args.poe_var and args.nobs and args.time and args.delta_t and args.outdir:
    # spl = SplineLearnerPOE_4D(use_mm=args.use_mm, a=args.aval, b=args.bval, num_bact=args.num_bact,
    #                     MEAS_VAR=args.mvar, PROC_VAR=args.pvar, THETA_VAR=args.theta_var, AVAR=args.avar,
    #                     BVAR=args.bvar, POE_VAR=args.poe_var, NSAMPS=args.nobs, TIME=args.time,
    #                     DT=args.delta_t,outdir = args.outdir)
    # elif args.use_mm and args.aval and args.outdir:
    spl = SplineLearnerPOE_4D(
        use_mm=args.use_mm, a=args.aval, outdir=args.outdir, bypass_f1=args.useF1)
    with open(args.outdir + '_class', 'wb') as f:
        pickle.dump(spl, f)
    spl.run(gibbs_steps=args.gibbs_steps)

if __name__ == "__main__":
   main()
