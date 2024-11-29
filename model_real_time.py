import numpy as np
import pickle
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', dest='classifier', type=str, default='lr',
                        choices=['lr', 'svm', 'rf', 'knn', 'nn'])
    parser.add_argument('--COUGH', dest='cough', type=int, default=0, choices=[0,1])
    parser.add_argument('--MUSCLE_ACHES', dest='muscle_aches', type=int, default=0, choices=[0,1])
    parser.add_argument('--TIREDNESS', dest='tiredness', type=int, default=0, choices=[0,1])
    parser.add_argument('--SORE_THROAT', dest='sore_throat', type=int, default=0, choices=[0,1])
    parser.add_argument('--RUNNY_NOSE', dest='runny_nose', type=int, default=0, choices=[0,1])
    parser.add_argument('--STUFFY_NOSE', dest='stuffy_nose', type=int, default=0, choices=[0,1])
    parser.add_argument('--FEVER', dest='fever', type=int, default=0, choices=[0,1])
    parser.add_argument('--DIARRHEA', dest='diarrhea', type=int, default=0, choices=[0,1])
    parser.add_argument('--DIFFICULTY_BREATHING', dest='difficulty_breathing', type=int, default=0, choices=[0,1])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # use user input to create test data instance
    X_test = np.zeros(9)
    X_test[0] = args.cough
    X_test[1] = args.muscle_aches
    X_test[2] = args.tiredness
    X_test[3] = args.sore_throat
    X_test[4] = args.runny_nose
    X_test[5] = args.stuffy_nose
    X_test[6] = args.fever
    X_test[7] = args.diarrhea
    X_test[8] = args.difficulty_breathing

    X_test = X_test.reshape(1,-1)

    lr_file_name = "lr_covid_flu_classifier.pkl"
    svm_file_name = "svm_covid_flu_classifier.pkl"
    rf_file_name = "rf_covid_flu_classifier.pkl"
    kNN_file_name = "knn_covid_flu_classifier.pkl"
    NN_file_name = "NN_covid_flu_classifier.pkl"

    model = None

    # choose model based on user input
    if args.classifier == 'lr':
        with open(lr_file_name, 'rb') as file:
            model = pickle.load(file)
    elif args.classifier == 'svm':
        with open(svm_file_name, 'rb') as file:
            model = pickle.load(file)
    elif args.classifier == 'rf':
        with open(rf_file_name, 'rb') as file:
            model = pickle.load(file)
    elif args.classifier == 'knn':
        with open(kNN_file_name, 'rb') as file:
            model = pickle.load(file)
    elif args.classifier == 'nn':
        with open(NN_file_name, 'rb') as file:
            model = pickle.load(file)

    predictions = model.predict(X_test)

    print(predictions[0])