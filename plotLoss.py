#Steven 08/08/2020 
import argparse 
import sys
import matplotlib.pyplot as plt

#----------------------------------------------
#usgae: python plotloss.py .\weights\trainMain.log
#----------------------------------------------

def getLoss(log_file,startIter=0,stopIter=None):
    numbers = {'1','2','3','4','5','6','7','8','9'}
    with open(log_file, 'r') as f:
        lines  = [line.rstrip("\n") for line in f.readlines()]
        
        iters = []
        loss = []
        accuracy=[]
        val_loss = []
        val_accuracy = []
        for line in lines:
            trainIterRes = line.split(' ')
            epoch = 0
            if trainIterRes[0] == 'Epoch' and trainIterRes[1][-1:]!=':':
                #print(line)
                str = trainIterRes[1]
                epoch = int(str[:str.find('/')])
                #print(line,' epoch=',epoch)
                if(epoch<startIter):
                    continue       
                if stopIter and  epoch > stopIter:
                    break
            
                iters.append(epoch)
           
            if trainIterRes[0] == '41/41' and trainIterRes[3] != 'ETA:':
                #print(line)
                #print('loss,acc=',trainIterRes[7],trainIterRes[10],'val_loss,val_acc=',trainIterRes[13],trainIterRes[16])
                loss.append(float(trainIterRes[7]))
                accuracy.append(float(trainIterRes[10]))
                
                val_loss.append(float(trainIterRes[13]))
                val_accuracy.append(float(trainIterRes[16]))

    return iters,loss,accuracy,val_loss,val_accuracy

def plotLoss(iters,loss,val_loss):
    ax = plt.subplot(1,1,1)
   
    #ax.set_title(name)
    ax.plot(iters,loss,label='On train set')
    ax.plot(iters,val_loss,label='On validation set')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    #plt.ylim(0, 4)
    #plt.yscale("log")
    #plt.legend()
    plt.show()
    
def plotAcc(iters,acc,val_acc):
    ax = plt.subplot(1,1,1)
   
    #ax.set_title(name)
    ax.plot(iters,acc,label='On train set')
    ax.plot(iters,val_acc,label='On validation set')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    #plt.ylim(0, 4)
    #plt.yscale("log")
    #plt.legend()
    plt.show()
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list', nargs='+', help='path to log file', required=True)
    parser.add_argument('-s', '--start', help = 'startIter')
    parser.add_argument('-t', '--stop', help = 'stopIter')
    
    return parser.parse_args()

def main():
    args = argCmdParse()
    
    startIter = 0
    stopIter = None
    if args.start:
        startIter = int(args.start)
    if args.stop:
        stopIter = int(args.stop)
        
    print(args.list,startIter,stopIter)
    file = args.list[0]
    iters,loss,acc,val_loss,val_acc = getLoss(file,startIter,stopIter)

    #print('loss=',loss)
    #print('val_loss=',val_loss)
    print('acc=',acc)
    print('val_acc=',val_acc)
    
    plotLoss(iters,loss,val_loss)
    plotAcc(iters,acc,val_acc)
    
if __name__ == "__main__":
    main()
    