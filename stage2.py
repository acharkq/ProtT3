import os
try:
    from model.opt_flash_attention import replace_opt_attn_with_flash_attn
except ModuleNotFoundError:
    pass
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from data_provider.stage2_dm import Stage2DM, Stage2MixDM
from data_provider.prot_qa_dm import ProtQADM
from model.blip2_stage2 import Blip2Stage2
from model.dist_funs import MyDeepSpeedStrategy
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)




def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args, map_location='cpu')
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    if args.mix_dataset:
        dm = Stage2MixDM(args.root, args)
    else:
        if args.root.find('PDBDataset') >= 0:
            dm = ProtQADM(args.root, args)
        else:
            dm = Stage2DM(args.root, args)
    dm.init_tokenizer(model.blip2.llm_tokenizer, model.blip2.plm_tokenizer)
    
    callbacks = []
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            find_unused_parameters = (not args.ptm) or (not args.lm)
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
        elif args.strategy == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            NotImplementedError()
    else:
        strategy = None
        args.devices = eval(args.devices)
    if args.use_wandb_logger:
        Path(f'./all_checkpoints/{args.filename}/wandb').mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=f'./all_checkpoints/{args.filename}/')
    else:
        logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        # limit_train_batches=2,
        # limit_val_batches=2,
    )
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)#, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy', type=str, default='deepspeed')

    # trainer arguments
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser.add_argument('--mix_dataset', action='store_true', default=False)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.enable_flash:
        replace_opt_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

