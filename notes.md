## nn.parallel.DataParallel()
1. before training: model=nn.parallel.DataParallel(model)
2. after step to save state_dict: model_=model.module
3. when eval bsize is none: model_=model(keep bsize have if use module)