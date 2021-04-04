import libs.parameters as parameters
import json
import sys
import os
import tools.surrogate_mapping as surrogate_mapping
from main import main
import shutil
import tools.evaluate as evalaute_one
import tools.evaluate_ensemble as evaluate_ens


def create_surrogate_mapping(data_dir, g_config, seed):
    dataset = g_config['dataset']
    surrogate_threshold = g_config['surrogate_threshold']
    arch = g_config['arch']
    tmp_model_dir = os.path.join(
        data_dir, dataset, f'deepxml.{arch}', f"{surrogate_threshold}.{seed}")
    data_dir = os.path.join(data_dir, dataset)
    try:
        os.makedirs(tmp_model_dir, exist_ok=False)
        surrogate_mapping.run(
            feat_fname=os.path.join(data_dir, g_config["trn_feat_fname"]),
            lbl_fname=os.path.join(data_dir, g_config["trn_label_fname"]),
            feature_type=g_config["feature_type"],
            method=g_config['surrogate_method'],
            threshold=g_config['surrogate_threshold'],
            seed=seed,
            tmp_dir=tmp_model_dir)
    except FileExistsError:
        print("Using existing data for surrogate task!")
    finally:
        data_stats = json.load(
            open(os.path.join(tmp_model_dir, "data_stats.json")))
        mapping = os.path.join(
            tmp_model_dir, 'surrogate_mapping.txt')
    return data_stats, mapping


def evaluate(g_config, data_dir, pred_fname, filter_fname=None, betas=-1, n_learners=1):
    if n_learners == 1:
        func = evalaute_one.main
    else:
        func = evaluate_ens.main

    dataset = g_config['dataset']
    data_dir = os.path.join(data_dir, dataset)
    A = g_config['A']
    B = g_config['B']
    if 'save_top_k' in g_config:
        top_k = g_config['save_top_k']
    else:
        top_k = g_config['top_k']
    ans = func(
        tst_label_fname=os.path.join(
            data_dir, g_config["tst_label_fname"]),
        trn_label_fname=os.path.join(
            data_dir, g_config["trn_label_fname"]),
        pred_fname=pred_fname,
        A=A, 
        B=B,
        top_k=top_k,
        filter_fname=filter_fname, 
        betas=betas, 
        save=g_config["save_predictions"])
    return ans


def print_run_stats(train_time, model_size, avg_prediction_time, fname=None):
    line = "-"*30 
    out = f"Training time (sec): {train_time:.2f}\n"
    out += f"Model size (MB): {model_size:.2f}\n"
    out += f"Avg. Prediction time (msec): {avg_prediction_time:.2f}"
    out = f"\n\n{line}\n{out}\n{line}\n\n"
    print(out)
    if fname is not None:
        with open(fname, "a") as fp:
            fp.write(out)


def run_deepxml(work_dir, version, seed, config):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']
    use_reranker = g_config['use_reranker']

    # run stats
    train_time = 0
    model_size = 0
    avg_prediction_time = 0

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')

    filter_fname = os.path.join(data_dir, dataset, 'filter_labels_test.txt')
    if not os.path.isfile(filter_fname):
        filter_fname = None
    
    result_dir = os.path.join(
        work_dir, 'results', 'DeepXML', arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', 'DeepXML', arch, dataset, f'v_{version}')
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['surrogate'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'surrogate')
    args.result_dir = os.path.join(result_dir, 'surrogate')

    # Create the label mapping for classification surrogate task
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        data_dir, g_config, seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train intermediate representation
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['surrogate'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    _train_time, _ = main(args)
    train_time += _train_time

    # performance on surrogate task
    args.mode = 'predict'
    main(args)

    # train final representation and extreme classifiers
    _args.update(config['extreme'])
    args = _args.params
    args.surrogate_mapping = None
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    _train_time, _model_size = main(args)
    train_time += _train_time
    model_size += _model_size

    # predict using extreme classifiers
    args.pred_fname = 'tst_predictions'
    args.mode = 'predict'
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time

    # copy the prediction files to level-1
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    # evaluate
    pred_fname = os.path.join(result_dir, 'tst_predictions')
    ans = evaluate(
        g_config=g_config,
        data_dir=data_dir,
        pred_fname=pred_fname,
        filter_fname=filter_fname,
        betas=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90])
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, "w") as fp:
        fp.write(ans)

    # run re-ranker
    if use_reranker:
        args.get_only = 'clf'
        args.tst_feat_fname = args.trn_feat_fname
        args.tst_label_fname = args.trn_label_fname
        args.pred_fname = "trn_predictions"

        os.makedirs(os.path.join(result_dir, 'reranker'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'reranker'), exist_ok=True)

        # predict on train set
        _, _train_time, _ = main(args)
        train_time += _train_time
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'trn_predictions_clf.npz'),
            os.path.join(model_dir, 'reranker', 'trn_shortlist.npz'))
        shutil.copy(
            os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
            os.path.join(model_dir, 'reranker', 'tst_shortlist.npz'))

        _args.update(config['global'])
        _args.update(config['reranker'])
        args = _args.params
        args.num_nbrs = args.top_k
        args.model_dir = os.path.join(model_dir, 'reranker')
        args.result_dir = os.path.join(result_dir, 'reranker')

        args.mode = 'train'
        args.arch = os.path.join(os.getcwd(), f'{arch}.json')
        temp = data_stats['extreme'].split(",")
        args.num_labels = int(temp[1])
        args.vocabulary_dims = int(temp[0])
        _train_time, _model_size = main(args)
        train_time += _train_time
        model_size += _model_size


        # re-rank the predictions using the re-ranker
        args.mode = 'predict'
        args.get_only = 'ens'
        args.pred_fname = 'tst_predictions_reranker'
        _, _ , _pred_time = main(args)
        avg_prediction_time += _pred_time
        shutil.copy(
            os.path.join(result_dir, 'reranker',
                        'tst_predictions_reranker_ens.npz'),
            os.path.join(result_dir, 'tst_predictions_reranker_clf.npz'))

        shutil.copy(
            os.path.join(result_dir, 'tst_predictions_knn.npz'),
            os.path.join(result_dir, 'tst_predictions_reranker_knn.npz'))

        pred_fname = os.path.join(
            result_dir, f'tst_predictions_reranker')
        ans = evaluate(
            g_config=g_config,
            data_dir=data_dir,
            filter_fname=filter_fname,
            pred_fname=pred_fname,
            betas=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90])
        with open(f_rstats, 'a') as fp:
            fp.write("\nRe-ranker\n\n")
            fp.write(ans)
    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, f"score_{g_config['beta']:.2f}.npz"), \
        train_time, model_size, avg_prediction_time


def run_deepxml_ova(work_dir, version, seed, config):
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')
    result_dir = os.path.join(
        work_dir, 'results', 'DeepXML-OVA', arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', 'DeepXML-OVA', arch, dataset, f'v_{version}')
    filter_fname = os.path.join(data_dir, dataset, 'filter_labels_test.txt')
    if not os.path.isfile(filter_fname):
        filter_fname = None

    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['extreme'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)


    data_stats, _ = create_surrogate_mapping(
        data_dir, g_config, seed)

    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    train_time, model_size = main(args)

    # predict using extreme classifiers
    args.pred_fname = 'tst_predictions'
    args.mode = 'predict'
    _, _ , avg_prediction_time = main(args)

    # copy the prediction files to level-1
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions.npz'),
        os.path.join(result_dir, 'tst_predictions.npz'))
    pred_fname = os.path.join(result_dir, 'tst_predictions')
    ans = evaluate(
        g_config=g_config,
        filter_fname=filter_fname,
        data_dir=data_dir,
        pred_fname=pred_fname)
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, 'w') as fp:
        fp.write(ans)
    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, "score.npz"), \
        train_time, model_size, avg_prediction_time


def run_deepxml_ann(work_dir, version, seed, config):
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')
    result_dir = os.path.join(
        work_dir, 'results', 'DeepXML-ANNS', arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', 'DeepXML-ANNS', arch, dataset, f'v_{version}')
    filter_fname = os.path.join(data_dir, dataset, 'filter_labels_test.txt')
    if not os.path.isfile(filter_fname):
        filter_fname = None
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['extreme'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_stats, _ = create_surrogate_mapping(
        data_dir, g_config, seed)

    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    train_time, model_size = main(args)

    # predict using extreme classifiers
    args.pred_fname = 'tst_predictions'
    args.mode = 'predict'
    _, _ , avg_prediction_time = main(args)

    # copy the prediction files to level-1
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    pred_fname = os.path.join(result_dir, 'tst_predictions')
    ans = evaluate(
        g_config=g_config,
        filter_fname=filter_fname,
        data_dir=data_dir,
        pred_fname=pred_fname,
        betas=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90])
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, 'w') as fp:
        fp.write(ans)
    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, f"score_{g_config['beta']:.2f}.npz"), \
        train_time, model_size, avg_prediction_time


def run_one(work_dir, model_type, version, seed, config):
    if model_type == "DeepXML":
        return run_deepxml(work_dir, version, seed, config)
    elif model_type == "DeepXML-OVA":
        return run_deepxml_ova(work_dir, version, seed, config)
    elif model_type == "DeepXML-ANNS":
        return run_deepxml_ann(work_dir, version, seed, config)
    else:
        raise NotImplementedError("Unknown model type!")


def run_ensemble(work_dir, model_type, version, seeds, config):
    train_time, model_size, avg_prediction_time = 0, 0, 0
    pred_fname = []
    for idx, seed in enumerate(seeds):
        print(f"Running learner {idx} with seed {seed}")
        f, tt, ms, pt = run_one(
            work_dir, model_type, f"{version}_{seed}", seed, config)        
        train_time += tt
        model_size += ms
        avg_prediction_time += pt
        pred_fname.append(f)
    arch = config["global"]['arch']
    dataset = config["global"]['dataset']
    result_dir = os.path.join(
        work_dir, 'results', model_type, arch, dataset, f'v_{version}_{seed}')
    print("Evaluating ensemble")
    ans = evaluate(
        g_config=config["global"],
        data_dir=os.path.join(work_dir, 'data'),
        pred_fname=pred_fname,
        n_learners=len(seeds))
    f_rstats = os.path.join(result_dir, "log_ens_eval.txt")
    with open(f_rstats, 'w') as fp:
        fp.write(ans)
    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)


if __name__ == "__main__":
    model_type = sys.argv[1]
    work_dir = sys.argv[2]
    version = sys.argv[3]
    config = sys.argv[4]
    seed = sys.argv[5]
    if "," in seed:
        seeds = list(map(int, seed.split(",")))
        run_ensemble(
            work_dir=work_dir,
            model_type=model_type,
            version=version,
            seeds=seeds,
            config=json.load(open(config)))
    else:
        seed = int(seed)
        run_one(
            work_dir=work_dir,
            model_type=model_type,
            version=f"{version}_{seed}",
            seed=seed,
            config=json.load(open(config)))
