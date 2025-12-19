import pandas as pd
import torch
from torch import nn
import models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_utils import save_model, load_model, weighted_accuracy, unweighted_accuracy, weighted_precision, unweighted_precision, write_csv_record
import sys
import os
import json
import csv
import torch.nn.functional as F

from datetime import datetime, timezone, timedelta

# 한국 시간대 (UTC+9)
kst = timezone(timedelta(hours=9))

# 현재 시간 (KST 기준)
now_kst = datetime.now(kst)

# 원하는 포맷으로 출력
formatted_time = str(now_kst.strftime('%Y%m%d%H%M%S'))

# print(formatted_time)

def initiate(hyp_params, train_loader, dev_loader, test_loader):
    tva_model = getattr(models, 'TVAModel_Self')(hyp_params)
    tva_model = tva_model.double().to('cuda')
    #import pdb
    #pdb.set_trace()
    optimizer = getattr(optim, hyp_params.optim)(filter(lambda p: p.requires_grad, tva_model.parameters()), lr=hyp_params.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'tva_model': tva_model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,
                }
    return train_model(settings, hyp_params, train_loader, dev_loader, test_loader)

### training and evaluation

def train_model(settings, hyp_params, train_loader, dev_loader, test_loader):
    tva_model = settings['tva_model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    def train(tva_model_, criterion_, optimizer_, uni_a_model, uni_v_model, uni_t_model):
        epoch_loss_total = 0
        tva_model_.train()
        for i_batch, batch_X in enumerate(train_loader):
            x_text, x_vid, x_audio, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3]
            # x_text, x_vid, vid_seqN, x_audio, x_pros, aud_seq, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3], batch_X[4], batch_X[5], batch_X[6]
            
            x_text = x_text.to('cuda').double()
            x_vid = x_vid.to('cuda').double()
            x_audio = x_audio.to('cuda').double()
            labels = labels.to('cuda')


            tva_model_.zero_grad()
            batch_size = x_text.size(0)
            
           
            
            if hyp_params.MKD:
                preds,  x_tva, x_aux_t, x_aux_v, x_aux_a = tva_model_(x_text.double(), x_vid.double(), x_audio.double())#(x_text, x_vid, vid_seqN, x_audio, x_pros, aud_seq)
                with torch.no_grad():
                    x_teacher_a_pred,_ = uni_a_model(x_text.double(), x_vid.double(), x_audio.double())
                    x_teacher_v_pred,_ = uni_v_model(x_text.double(), x_vid.double(), x_audio.double())
                    x_teacher_t_pred,_ = uni_t_model(x_text.double(), x_vid.double(), x_audio.double())
                
                x_teacher_a_pred = x_teacher_a_pred.detach()
                x_teacher_v_pred = x_teacher_v_pred.detach()
                x_teacher_t_pred = x_teacher_t_pred.detach()
                # print("labels 출력")
                # print(labels)
                
                aux_loss = 0
                MKD_loss = 0
                print("MKD 중")
         
                if 'a' in hyp_params.MKD_modals:
                    MKD_loss += F.kl_div(F.log_softmax(x_aux_a,dim=1), F.softmax(x_teacher_a_pred,dim=1), reduction="batchmean")
                elif 'v' in hyp_params.MKD_modals:
                    MKD_loss += F.kl_div(F.log_softmax(x_aux_v,dim=1), F.softmax(x_teacher_v_pred,dim=1), reduction="batchmean")
                elif 't' in hyp_params.MKD_modals:
                    MKD_loss += F.kl_div(F.log_softmax(x_aux_t,dim=1), F.softmax(x_teacher_t_pred,dim=1), reduction="batchmean")
                    
                if 'a' in hyp_params.MKD_modals:
                    aux_loss += criterion_(x_aux_a, labels)
                elif 'v' in hyp_params.MKD_modals:
                    aux_loss += criterion_(x_aux_v, labels)
                elif 't' in hyp_params.MKD_modals:
                    aux_loss += criterion_(x_aux_t, labels)
                    
                MKD_loss = MKD_loss + aux_loss
                
                MKD_loss.backward()
                
                optimizer_.step()
                
                
                
                
                tva_model_.zero_grad()
                
            tva_model_.zero_grad()
            preds,  x_tva, x_aux_t, x_aux_v, x_aux_a = tva_model_(x_text.double(), x_vid.double(), x_audio.double())
            raw_loss = criterion_(preds, labels)
            raw_loss.backward()
            #torch.nn.utils.clip_grad_norm_(tva_model_.parameters(), 0.01)
            optimizer_.step()
            epoch_loss_total += raw_loss.item() * batch_size
            if i_batch % 20 == 0:
                print(f"  🌀 [Epoch {epoch} | Batch {i_batch}/{len(train_loader)}] Loss: {raw_loss.item():.4f}")
        return epoch_loss_total / hyp_params.n_train

    def evaluate(tva_model_, criterion_, test=False):
        tva_model_.eval()
        loader = test_loader if test else dev_loader
        total_loss = 0.0
        results_ = []
        truths_ = []
        ints_ = [] # intermediate embeddings
        with torch.no_grad():
            for i_batch, batch_X in enumerate(loader):
                x_text, x_vid, x_audio, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3]


                x_text = x_text.to('cuda').double()
                x_vid = x_vid.to('cuda').double()
                x_audio = x_audio.to('cuda').double()
                labels = labels.to('cuda')


                # x_text, x_vid, vid_seqN, x_audio, x_pros, aud_seq, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3], batch_X[4], batch_X[5], batch_X[6]
                batch_size = x_text.size(0)
                preds, x_tva, x_aux_t, x_aux_v, x_aux_a = tva_model_(x_text.double(), x_vid.double(), x_audio.double())#(x_text, x_vid, vid_seqN, x_audio, x_pros, aud_seq)
                total_loss += criterion_(preds, labels).item() * batch_size
                # Collect the results into dictionary
                results_.append(preds)
                truths_.append(labels)
                ints_.append(x_tva)
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_dev)

        results_ = torch.cat(results_)
        truths_ = torch.cat(truths_)
        ints_ = torch.cat(ints_)
        return avg_loss, results_, truths_, ints_

    def perf_eval(results_, truths_):
        results_ = torch.argmax(results_, dim=1)
        truths_ = truths_.cpu().numpy()
        results_ = results_.cpu().numpy()
        results_ = results_.tolist()
        truths_ = truths_.tolist()
        wa_ = weighted_accuracy(truths_, results_)
        uwa_ = unweighted_accuracy(truths_, results_)
        wp_ = weighted_precision(truths_, results_)
        uwp_ = unweighted_precision(truths_, results_)
        return wa_, uwa_, wp_, uwp_


    # best_valid = 1e8
    best_val_uwa = 0
    es = 0
    #"""
    print("몇번째 폴더일까?")
    print(hyp_params.folder)
    print(type(hyp_params.folder))
    
    if hyp_params.MKD:
        import pandas as pd

        UNI_CSV = "/workspace/MKDMAMURI/MKD_0718_MKD_yoong/MKD/best_unimodals.csv"
        UNIMODAL_PATH_CSV = pd.read_csv(UNI_CSV)

        # folder 타입 통일: 문자열이든 숫자든 동일하게 비교되도록
        folder_value = int(hyp_params.folder) if isinstance(hyp_params.folder, str) else hyp_params.folder

        uni_a_model = uni_v_model = uni_t_model = None

        if 'a' in hyp_params.MKD_modals:
            df_a = UNIMODAL_PATH_CSV[
                (UNIMODAL_PATH_CSV['folder'] == folder_value) &
                (UNIMODAL_PATH_CSV['modals'] == 'a') &
                (UNIMODAL_PATH_CSV['normalization'] == hyp_params.normalization)
            ]
            if not df_a.empty:
                path = df_a["saved_pth"].values[0]+"/final_exp.pth"
                print("[A] uni_a_teacher_path:", path)
                uni_a_model = load_model(name=path)

        if 'v' in hyp_params.MKD_modals:
            df_v = UNIMODAL_PATH_CSV[
                (UNIMODAL_PATH_CSV['folder'] == folder_value) &
                (UNIMODAL_PATH_CSV['modals'] == 'v') &
                (UNIMODAL_PATH_CSV['normalization'] == hyp_params.normalization)
            ]
            if not df_v.empty:
                path = df_v["saved_pth"].values[0]+"/final_exp.pth"
                print("[V] uni_v_teacher_path:", path)
                uni_v_model = load_model(name=path)

        if 't' in hyp_params.MKD_modals:
            df_t = UNIMODAL_PATH_CSV[
                (UNIMODAL_PATH_CSV['folder'] == folder_value) &
                (UNIMODAL_PATH_CSV['modals'] == 't') &
                (UNIMODAL_PATH_CSV['normalization'] == hyp_params.normalization)
            ]
            if not df_t.empty:
                path = df_t["saved_pth"].values[0]+"/final_exp.pth"
                print("[T] uni_t_teacher_path:", path)
                uni_t_model = load_model(name=path)

        print("MKD: 모델 불러오기 완료")

        # 모델이 로드되었을 경우만 .eval() 수행
        if uni_a_model: uni_a_model.eval()
        if uni_v_model: uni_v_model.eval()
        if uni_t_model: uni_t_model.eval()
        
    else:
        pass
    # else: hyp_params.normalization:
    #     if hyp_params.folder=="01":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="02":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="03":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="04":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="05":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    # else:
    #     if hyp_params.folder=="01":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="02":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="03":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="04":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"
    #     elif hyp_params.folder=="05":
    #         uni_a_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed89587_20250718201209/final_exp.pth"
    #         uni_v_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed51909_20250718200708/final_exp.pth"
    #         uni_t_teacher_path = "/workspace/MKDMAMURI/models_MKD/fold04/seed38850_20250718203736/final_exp.pth"        
    
    
    pth_dir_path ="/workspace/MKDMAMURI/models_MKD"
    folders_path = os.path.join(pth_dir_path, f"fold{hyp_params.folder}", f"seed{hyp_params.seed}_{formatted_time}")
    results_csv_path = os.path.join(folders_path, "results.csv")
    for epoch in range(1, hyp_params.num_epochs + 1):
        train_total_loss = train(tva_model, criterion, optimizer, uni_a_model, uni_v_model, uni_t_model)
        val_loss, val_res, val_tru, _ = evaluate(tva_model, criterion, test=False)
        val_wa, val_uwa, val_wp, val_uwp = perf_eval(val_res, val_tru)
        test_loss, tst_res, tst_tru, _ = evaluate(tva_model, criterion, test=True)
        tst_wa, tst_uwa, tst_wp, tst_uwp = perf_eval(tst_res, tst_tru)
        scheduler.step(val_loss)  # Decay learning rate by validation loss
        print("-" * 50)
        print('Epoch {:2d} | Train Total Loss {:5.4f}'.format(epoch, train_total_loss))
        print('Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(val_loss, test_loss))
        print('Valid WA {:5.4f} | UWA {:5.4f} | WP {:5.4f} | UWP {:5.4f}'.format(val_wa, val_uwa, val_wp, val_uwp))
        print('Test WA {:5.4f} | UWA {:5.4f} | WP {:5.4f} | UWP {:5.4f}'.format(tst_wa, tst_uwa, tst_wp, tst_uwp))
        print("-" * 50)
        
        params_dict = vars(hyp_params)
        epoch_info = {'epoch': epoch}
        val_metrics = {
            "wa": val_wa,
            "uwa": val_uwa,
            "wp": val_wp,
            "uwp": val_uwp,
        }

        record = {**params_dict, **val_metrics, **epoch_info, **{"time": formatted_time}, **{"saved_pth": folders_path}}
        
    
        if val_uwa > best_val_uwa:
            print("Saved model at epoch: ", epoch)

            
            os.makedirs(name=folders_path, exist_ok=True)
            
            # ✅ settings 저장
            settings_path = os.path.join(folders_path, "settings.json")
            with open(settings_path, 'w') as f:
                json.dump(vars(hyp_params), f, indent=4)

            model_path = os.path.join(folders_path, "final_exp.pth")
            save_model(tva_model, name=model_path)
            
            best_val_uwa = val_uwa
            es = 0
        else:
            es = es + 1
            if es >= 10:
                break
        csv_path = os.path.join(results_csv_path)
        write_csv_record(csv_path, record)
        
        
    model = load_model(name=model_path)
    _, results, truths, ints = evaluate(model, criterion, test=True)
    results = torch.argmax(results, dim=1)
    truths = truths.cpu().numpy()
    results = results.cpu().numpy()

    from sklearn.metrics import classification_report as cr
    print(cr(truths, results))
    from sklearn.metrics import confusion_matrix as cm
    print(cm(truths, results))

    results = results.tolist()
    truths = truths.tolist()
    wa = weighted_accuracy(truths, results)
    uwa = unweighted_accuracy(truths, results)
    wp = weighted_precision(truths, results)
    uwp = unweighted_precision(truths, results)
    print("weighted accuracy:", wa)
    print("unweighted accuracy:", uwa)
    print("weighted precision:", wp)
    print("unweighted precision:", uwp)
    
    

    params_dict = vars(hyp_params)
    metrics = {
        "wa": wa,
        "uwa": uwa,
        "wp": wp,
        "uwp": uwp,
    }
    
    record = {**vars(hyp_params), **metrics, **{'epoch': 'final'}, **{"time": formatted_time},**{"saved_pth": folders_path}}
   
    write_csv_record(results_csv_path, record)
  