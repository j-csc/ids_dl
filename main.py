from model_builder import modelBldr as mdbldr
from data_preprocessor import preprocessor as prep
import joblib
import torch
import autoencoder as ae
    
def main():
  # Preprocessing data

  # preprocessor = prep()
  # preprocessor.load_data_bulk()
  # preprocessor.preprocessing()

  # Model building

  # model = mdbldr()
  # mdl, y_pred = model.random_forest_model()
  # model.eval_metrics(mdl, y_pred)
  # model.saveModel(mdl, "Random_Forest_model")

  # Model loading and training

  # model = joblib.load('./saved_models/ID3_decision_tree_model.joblib')
  # mdb = mdbldr(downsample=True)
  # y_pred = model.predict(mdb.X_test)
  # mdb.eval_metrics(model, y_pred)

  # Pytorch model loading and training

  # model = torch.load('./saved_models/Autoencoder.pt')
  # model.eval()
  # print(model)
  # ae.predict(model)

  pass
main()