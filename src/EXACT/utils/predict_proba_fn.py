def predict_proba(X, model):
        import torch
        import torch.nn.functional as F
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Predict proba called, shape: ",X.shape)

        model.eval()

        #Convert numpy -> torch

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X,dtype=torch.float32)

        # Handling the error of height,width,channel by converting the height,width,channel -> channel, height, width
        # Pytorch model expects input shape [batch,channels,height,widht] but
        # Lime is sending image as [batch,height,width,channels]
        # So when lime perturbs image it passes in this format to our pytorch wrapper and it will crash because of unexpected input type
        # Handling error ->
        if X.ndim == 4 and X.shape[-1] == 3:
            # (N, H, W, C) -> (N, C, H, W)
            X = X.permute(0,3,1,2) # Permute is use to change the dimension 
        

        X = X.to(device)

        with torch.no_grad():
            logits = model(X)

            # Binary vs multi-class
            if logits.ndim == 2 and logits.shape[-1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits,dim=1)
        
        return probs.cpu().numpy()