from tkinter.tix import Y_REGION
from utils.functions import EarlyStopping
from utils.train import training
from utils.validation import evaluation
import hydra
from utils.functions import save_plot
import wandb

def run_fold(max_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, fold, cfg):

    model.to(device)
    early_stopping = EarlyStopping(patience=cfg.early_stopping.patience, verbose=True, delta=0, path=f'best_param_of_cv_{fold}.pt', trace_func=logger.info)
    train_losses = []
    train_scores = []
    val_losses = []
    val_scores = []

    

    for epoch in range(max_epochs):
        logger.info(f'=============== epoch:{epoch+1}/{max_epochs} ===============')

        #scheduler.step(epoch)
        ########################
        #        train         #
        ########################
        #logger.info('------------- start of training ------------')
        train_loss, train_score = training(model, train_loader, criterion, optimizer, scheduler, device, logger)
        train_losses.append(train_loss)
        train_scores.append(train_score)

        ########################
        #      evaluation      #
        ########################
        #logger.info('------------- start of evaluation ------------')
        val_loss, val_score = evaluation(model, val_loader, criterion, device, logger)
        val_losses.append(val_loss)
        val_scores.append(val_score)

        logger.info(f'[result of epoch {epoch+1}/{max_epochs}]')
        logger.info(f'lr:{optimizer.param_groups[0]["lr"]}')
        logger.info(f'train_loss:{train_loss} train_score:{train_score}')
        logger.info(f'val_loss:{val_loss} val_score:{val_score}')

        ########################
        #     early stopping   #
        ########################
        is_best = early_stopping(val_score, model, check_loss=False)
        if is_best:
            best_val_score = val_score
        if early_stopping.early_stop:
            logger.info("early stopping is adopted.")
            break
    
        if not cfg.DEBUG:
            wandb.log({
                f"train_loss_{fold}": train_loss,
                f"val_loss_{fold}": val_loss,
                f"val_score_{fold}": val_score
                })


    # lossのlogグラフを画像にして保存
    save_plot({"train_loss":train_losses, "val_loss":val_losses}, f"loss_log_{fold}.jpg", title=f"loss_log_{fold}", xlabel="epoch", ylabel="loss")
    save_plot({"train_score":train_scores, "val_score":val_scores}, f"score_log_{fold}.jpg", title=f"score_log_{fold}", xlabel="epoch", ylabel="score")
    print(f"best_score_of_cv_{fold}: {best_val_score}")
    return best_val_score