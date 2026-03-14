def predict_proba(X, model, mode = "classification"):
    import torch

    #-------------------- Device --------------------
    device = next(model.parameters()).device
    model.eval()

    # -------------------- Handle dictionary inputs --------------------
    # For transformer tokenizers
    if isinstance(X, dict):

        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in X.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)
        # -------------------- Handle Tensor inputs--------------------
    else:
        # -------------------- Convert to Tensor --------------------
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        # -------------------- Image handling --------------------
        # Case : Lime imager gives (N, H, W, C)
        if X.ndim == 4 and X.shape[-1] in (1,3):
            # Convert N,H,W,C - > N, C, H, W
            # Handling the error of height,width,channel by converting the height,width,channel -> channel, height, width
            # Pytorch model expects input shape [batch,channels,height,widht] but
            # Lime is sending image as [batch,height,width,channels]
            # So when lime perturbs image it passes in this format to our pytorch wrapper and it will crash because of unexpected input type
            # Handling error ->

            X = X.permute(0, 3, 1, 2).float() # Permute is use to change the dimension

        # -------------------- TEXT (token IDs)  --------------------
        elif X.dtype in (torch.int32 , torch.int64):
            X = X.long()

        # -------------------- TABULAR / EMBEDDINGS  --------------------
        else:
            X = X.float()
        
        X = X.to(device)

        # -------------------- Forward pass --------------------
        with torch.no_grad():
            outputs = model(X)
    # -------------------- Extract logits --------------------
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    
    elif hasattr(outputs, "logits"):
        logits = outputs.logits

    else:
        raise ValueError("Model output format not supported ")
    
    # -------------------- Regression --------------------
    if mode == "regression":
        return logits.detach().cpu().numpy()
    
    # -------------------- Convert logits → probabilities --------------------

    # Binary Classification 
    if logits.ndim == 2 and logits.shape[-1] == 1:
        probs_pos = torch.sigmoid(logits)

        # Convert (N,1) -> (N,2)
        probs = torch.cat([1 - probs_pos, probs_pos], dim=1)

    # Mulit-class Classification
    else:
        probs = torch.softmax(logits, dim = 1)
        
    return probs.detach().cpu().numpy()