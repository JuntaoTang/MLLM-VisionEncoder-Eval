"""Independent MLLM entry; interpreter selection is owned by the orchestrator."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['continuous', 'discrete'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train-config')
    parser.add_argument('--data-config')
    parser.add_argument('--recipe')
    parser.add_argument('--finetune-tag')
    parser.add_argument('--eval-model')
    parser.add_argument('--stages', nargs='+', required=True)
    args = parser.parse_args()
    from vision_encoder_eval.mllm.utils.config import install_offline_hf_env
    install_offline_hf_env()
    from vision_encoder_eval.mllm.runner.pipeline import run_pipeline
    return run_pipeline(config_path=args.config, train_config_path=args.train_config,
                        data_config_path=args.data_config, mode=args.mode,
                        stages_override=args.stages, eval_model=args.eval_model,
                        recipe_override=args.recipe, finetune_tag_override=args.finetune_tag)

if __name__ == '__main__':
    raise SystemExit(main())
