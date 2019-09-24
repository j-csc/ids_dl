from model_builder import modelBldr as mdbldr

def main():
  model = mdbldr()
  model.load_data_bulk()
  model.preprocessing()
  pass

main()