from model_builder import modelBldr as mdbldr
from data_preprocessor import preprocessor as prep
def main():
  # preprocessor = prep()
  # preprocessor.load_data_bulk()
  # preprocessor.preprocessing()
  model = mdbldr()
  mdl, y_pred = model.random_forest_model()
  model.eval_metrics(mdl, y_pred)
  model.saveModel(mdl, "Random_Forest_model")
  pass
main()