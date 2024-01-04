from source import experiment, inference  # noqa: F401

if __name__ == "__main__":
    # common settings
    ex_num = 0
    dry_run = True
    local = True

    # experiment
    experiment.run(ex_num, dry_run=dry_run, local=local)

    # inference settings
    ignore_exception = False
    # inference
    inference.run(
        ex_num,
        local=True,
        dry_run=dry_run,
        ignore_exception=ignore_exception,
    )

    # Note: 以下を単体で走らせる場合に必要
    from source.abs.config import set_config

    set_config(local=local, experiment=False)

    # validate settings
    id_columns = ["row_id"]
    # validate prediction
    inference._validate_prediction(ex_num, id_columns=id_columns)

    # upload dataset
    inference._upload_dataset(ex_num)
