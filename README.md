# CS6910_DeepLearning_Assignment_3

# Deep Learning Assignment 3: RNN

##### ANIKET KESHRI CS23M013
This is assignment number 3 in the course, Fundamentals of Deep Learning CS6910 by Prof. Mitesh Khapra  at IIT madras. This assignment is based on the RNN with Aksharantar dataset released by AI4Bharat containing various languages dataset , i selected Hindi dataset containig 4096 words.

I run upto 200+ different configurations on "Hindi Dataset" ( 127 without attention and 100 with Attention) and track them all using wandb, we then find correlations with the best features and tune further searches to attempt to reach as high an accuracy as possible:-

Report can be accessed here:- https://wandb.ai/cs23m013/DL_A3/reports/Assignment-3--Vmlldzo3NzMyNDg3
#Libraries used : 
The code is written in Python and using notebook using following libraries used:
- torch
- numpy
- os
- wandb
- torchvision
- matplotlib
- ob


### Dataset
This assignment is based on the RNN with Aksharantar dataset released by AI4Bharat containig 4096 words.

For the hyper parameter optimisation stage i ran it in without model and with attention model than recorder the performance.

Once the best configuration is identified with the help of wandb using Random search or Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. 

### Language i choosed : 
  HINDI DATASET

### Recurrent neural network (RNN) 
It is a deep learning model that is trained to process and convert a sequential data input into a specific sequential data output. Sequential data is data—such as words, sentences, or time-series data—where sequential components interrelate based on complex semantics and syntax rules. An RNN is a software system that consists of many interconnected components mimicking how humans perform sequential data conversions, such as translating text from one language to another. RNNs are largely being replaced by transformer-based artificial intelligence (AI) and large language models (LLM), which are much more efficient in sequential data processing.

### classes involved
- class data_preprocess(Dataset): Class for making tokenized data iteratable"
- class encoder : precessing Encoder part of the model
- clasas decoder : processing Decoder part of the model
- class seq2seq : connecting encoder and decoder to generate the output

### cell Type used :
 - LSTM 
 - GRU
 - RNN

# Vanilla Model (Without Attention)  
Here I , built a RNN model without using the attention mechanish , i ran this total of 127 runs and recoderd the accuracy in wandb.

i used following hyper parameters configuration for this:

    sweep_configuration={
    'name':'cs23m013',   # Metric to optimize (e.g., validation accuracy)
    'method':'bayes',
    'metric':{'name':'val_acc','goal':'maximize'},
    'parameters':{

        'epochs':{
            'values':[1]  # List of values for the number of epochs (e.g., [1])
        },

         'learning_rate':{
            'values':[0.001 , 0.0001 , 0.00001]   # List of values for learning rate
        },

        'embedding_size':{
            'values':[64,128,256]   # List of values for embedding size
            },

        'num_encoder_layer':{  # List of values for the number of encoder layers
            'values':[1,2,3]
            },

        'num_decoder_layer':{  # List of values for the number of decoder layers
            'values':[1,2,3]
        },

        'hidden_layer_size':{
            'values':[64,128 , 256,512] # List of values for hidden layer size
            },

        'cell_type':{
            'values':['LSTM','GRU', 'RNN'] # List of values for RNN cell type (e.g., 'GRU')
            },

        'dropout':{
            'values':[0, 0.2,0.3 , 0.4 , 0.6]
            },

        'bidirection':{
            'values':[True,False]
            }

        }
    }

### Training     
Training part is done selecting one of the languages among all provider in dataset , i choosed Hindi dataset and trained the model on the Training dataset provided in the dataset.

### Functions used 
- generate_token :  #source: input column of csv file , target: target column of csv file
- def data_cleaning(text): # Remove punctuations and digits and normalizing the data
- forward_encoder : fordard part for enoder part
- forward_decoder : forward part for decoder part
- train_model : for training model
- evaluate : validation function
- main : Main training loop function

  
### How to run 'Train_Vanilla.py" file. 
**SELECT HINDI DATASET

    python Train_Vanilla.py -dpd /content/drive/MyDrive/DL/A3_DATA/aksharantar_sampled/hin -ep 1 -ct LSTM -ndl 3 -nel 3 
     
where ,  /content/drive/MyDrive/DL/A3_DATA/aksharantar_sampled/hin  is a data path for "Hindi" language 

### Defined arguments for hyperparameters (command for line arguments)
    parser.add_argument("-wp", '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard', type=str, default='DL_A3')
    parser.add_argument("-we", "--wandb_entity", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m013")
    parser.add_argument('-ep', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate used to optimize model parameters', type=float, default=0.001 , choices=[0.001 , 0.0001 ,  
    0.00001])
    parser.add_argument("-dp", "--dropout", default=0.2, type=float, choices=[0, 0.2,0.3 , 0.4 , 0.6])
    parser.add_argument("-lg", "--logger", type=bool, default=False, choices=[True, False], help="Log to wandb or not")
    parser.add_argument('-dpd', '--data_path_directory', help="Dataset", type=str, default='/content/drive/MyDrive/DL/A3_Data/aksharantar_sampled/hin')
    parser.add_argument("-bd", "--bidirection",type=bool,default=False, choices=[True, False],help="enter correct value of bidirection")
    parser.add_argument('-ct', '--cell_type', help='choices: ["LSTM", "GRU", "RNN"]', choices=["LSTM", "GRU", "RNN"], type=str, default='LSTM')
    parser.add_argument("-hls", "--hidden_layer_size", default=512, type=int, choices=[64,128 , 256,512])
    parser.add_argument("-ndl", "--num_decoder_layer", default=3,type=int, choices=[1,2,3])
    parser.add_argument("-nel", "--num_encoder_layer", default=3,type=int, choices=[1,2,3])
    parser.add_argument("-es", "--embedding_size", type=int, default=256 , choices=[64,128,256])


# Attention Model (using attention mechanish)  
Here I , built a RNN model using the attention mechanish , i ran this total of 100 runs and recoderd the accuracy in wandb.

i used following hyper parameters configuration for this:

    sweep_configuration={
    'name':'cs23m013',   # Metric to optimize (e.g., validation accuracy)
    'method':'bayes',
    'metric':{'name':'val_acc','goal':'maximize'},
    'parameters':{

        'epochs':{
            'values':[1]  # List of values for the number of epochs (e.g., [1])
        },

         'learning_rate':{
            'values':[0.001 , 0.0001 , 0.00001]   # List of values for learning rate
        },

        'embedding_size':{
            'values':[64,128,256]   # List of values for embedding size
            },

        'num_encoder_layer':{  # List of values for the number of encoder layers
            'values':[1,2,3]
            },

        'num_decoder_layer':{  # List of values for the number of decoder layers
            'values':[1,2,3]
        },

        'hidden_layer_size':{
            'values':[64,128 , 256,512] # List of values for hidden layer size
            },

        'cell_type':{
            'values':['LSTM','GRU', 'RNN'] # List of values for RNN cell type (e.g., 'GRU')
            },

        'dropout':{
            'values':[0, 0.2,0.3 , 0.4 , 0.6]
            },

        'bidirection':{
            'values':[True,False]
            }

        }
    }

### Training     
Training part is done selecting one of the languages among all provider in dataset , i choosed Hindi dataset and trained the model on the Training dataset provided in the dataset.

### Functions used 
- generate_token :  #source: input column of csv file , target: target column of csv file
- def data_cleaning(text): # Remove punctuations and digits and normalizing the data
- forward_encoder : fordard part for enoder part
- forward_decoder : forward part for decoder part
- train_model : for training model
- evaluate : validation function
- main : Main training loop function

  
### How to run 'Train_Attention.py" file.
**SELECT HINDI DATASET

    python Train_Attention.py -dpd /content/drive/MyDrive/DL/A3_DATA/aksharantar_sampled/hin -ep 1 -ct LSTM -ndl 3 -nel 3 
     
where ,  /content/drive/MyDrive/DL/A3_DATA/aksharantar_sampled/hin  is a data path for hindi language 

### Defined arguments for hyperparameters (command for line arguments)
    parser.add_argument("-wp", '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard', type=str, default='DL_A3')
    parser.add_argument("-we", "--wandb_entity", type=str, help="Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs23m013")
    parser.add_argument('-ep', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate used to optimize model parameters', type=float, default=0.001 , choices=[0.001 , 0.0001 , 
    0.00001])
    parser.add_argument("-dp", "--dropout", default=0.2, type=float, choices=[0, 0.2,0.3 , 0.4 , 0.6])
    parser.add_argument("-lg", "--logger", type=bool, default=False, choices=[True, False], help="Log to wandb or not")
    parser.add_argument('-dpd', '--data_path_directory', help="Dataset", type=str, default='/content/drive/MyDrive/DL/A3_Data/aksharantar_sampled/hin')
    parser.add_argument("-bd", "--bidirection",type=bool,default=False, choices=[True, False],help="enter correct value of bidirection")
    parser.add_argument('-ct', '--cell_type', help='choices: ["LSTM", "GRU", "RNN"]', choices=["LSTM", "GRU", "RNN"], type=str, default='LSTM')
    parser.add_argument("-hls", "--hidden_layer_size", default=512, type=int, choices=[64,128 , 256,512])
    parser.add_argument("-ndl", "--num_decoder_layer", default=3,type=int, choices=[1,2,3])
    parser.add_argument("-nel", "--num_encoder_layer", default=3,type=int, choices=[1,2,3])
    parser.add_argument("-es", "--embedding_size", type=int, default=256 , choices=[64,128,256])



