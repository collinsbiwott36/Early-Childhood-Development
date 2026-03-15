import optuna

def tune_model(objective_fn, n_trials=50, direction="maximize", seed=42):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)
    return study
