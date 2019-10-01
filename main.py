from model_builder import modelBldr as mdbldr
from data_preprocessor import preprocessor as prep
import joblib
import torch
import autoencoder as ae
    
def main():
  # Preprocessing data
  ans = input("Would you like to load new data? Y/N \n")
  if ans == "Y":
    preprocessor = prep()
    raw_path = input("Please provide the path to the folder storing all the csv files processed by ISCX flowmeter: \n")
    preprocessor.load_data_bulk()
    preprocessor.preprocessing()

  # Model building
  ans = input("Would you like to train a new model? Y/N \n")
  if ans == "Y":
    model_type = input("Please specify the model you would like to build? \n Options: RF, LR, ID3, AE \n")

    model = mdbldr()
    if model_type == "RF":
      mdl, y_pred = model.random_forest_model()
      model.eval_metrics(mdl, y_pred)
      model.saveModel(mdl, "Random_Forest_model")
    elif model_type == "ID3":
      mdl, y_pred = model.id3_decision_tree_model()
      model.eval_metrics(mdl, y_pred)
      model.saveModel(mdl, "ID3_decision_tree_model")
    elif model_type == "AE":
      ae.train_encoder(save=False)
    else:
      mdl, y_pred = model.logistic_regression_model()
      model.eval_metrics(mdl, y_pred)
      model.saveModel(mdl, "Logistic_model")
  else:
    # Model loading and training
    ans = input("Would you like to load an ML model then? Y/N \n")
    if ans == "Y":
      path = input("Please specify the path where the model is store? \n")
      model = joblib.load(path)
      mdb = mdbldr(downsample=True)
      y_pred = model.predict(mdb.X_test)
      mdb.eval_metrics(model, y_pred)
    else:
      ans = input("Would you like to load a Deep Learning model then? Y/N \n")
      if ans == "Y":
        path = input("Please specify the path where the model is store? \n")
        model = torch.load(path)
        model.eval()
        data_q = input("Do you have a file you want to run ? Y/N \n")
        if data_q == "Y":
          data_path = input("Please enter the path of the file: \n")
          ae.predict(model, df_path=data_path)
        else:
          ae.predict(model)
  
  print("Thanks for using this program! ")

  pass
main()