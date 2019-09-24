from model_builder import modelBldr as mdbldr
from data_preprocessor import preprocessor as prep
def main():
  # preprocessor = prep()
  # preprocessor.load_data_bulk()
  # preprocessor.preprocessing()
  model = mdbldr()
  lr_mdl, y_pred = model.logistic_regression_model()
  model.eval_metrics(lr_mdl, y_pred)

  pass

main()