from source import experiment, inference

if __name__ == "__main__":
    # common settings
    ex_num = 0
    dry_run = True
    local = True
    
    # experiment
    experiment.run(ex_num, dry_run=dry_run, local=local)
    
    # inference
    ignore_exception = False
    id_columns = ["not_implement_error"]
    inference.run(
        ex_num,
        local=True,
        dry_run=dry_run,
        ignore_exception=ignore_exception,
    )
    
    # validate prediction
    inference._validate_prediction(ex_num, id_columns=id_columns)

    # upload dataset
    inference._upload_dataset(ex_num)
