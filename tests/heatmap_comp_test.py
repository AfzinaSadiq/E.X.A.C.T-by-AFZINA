import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np
from EXACT.explainers import GradCAM, LimeExplainer_Image, IGImageExplainer
from EXACT.comparators import HeatmapComparator

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img    = Image.open("models/Te-me_0010.jpg").convert("RGB")
    img_np = np.array(Image.open("models/Te-me_0010.jpg"))
    p_img  = tf(img).unsqueeze(0).to(device)

    class tumor_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 4)
            ).to(device)
        def forward(self, x):
            return self.layer_stack(x)

    test_model = tumor_model()
    test_model.load_state_dict(torch.load("models/model_1.pth", map_location=device))

    # ── GradCAM family ───────────────────────────────────────────────
    explainer = GradCAM(test_model)
    gradcam_result   = explainer.explain(input_tensor=p_img, input_image=img_np, method="gradcam",   save_png=True)
    gradcampp_result = explainer.explain(input_tensor=p_img, input_image=img_np, method="gradcam++", save_png=True)
    xgradcam_result  = explainer.explain(input_tensor=p_img, input_image=img_np, method="xgradcam",  save_png=True)
    eigencam_result  = explainer.explain(input_tensor=p_img, input_image=img_np, method="eigencam",  save_png=True)

    # ── LIME ─────────────────────────────────────────────────────────
    lime_explainer = LimeExplainer_Image(model=test_model, target_size=(128, 128))
    lime_result    = lime_explainer.explain(image=img_np)

    # ── Integrated Gradients ─────────────────────────────────────────
    # ig_explainer = IGImageExplainer(model=test_model)
    # ig_result    = ig_explainer.explain(
    #     input_tensor=p_img,
    #     input_image=img_np,
    #     steps=100,       # 100 is fast enough for testing; raise to 200 for production
    #     save_png=True,
    # )

    # ── Compare ──────────────────────────────────────────────────────
    cmp = HeatmapComparator(model=test_model, device=device, deletion_steps=10, faithfulness_enabled=True)

    results = cmp.compare(
        entries={
            # (result_dict,  explainer_object,  extra_kwargs_for_stability_reruns)
            "GradCAM":   (gradcam_result,   explainer,      {"method": "gradcam"}),
            "GradCAM++": (gradcampp_result, explainer,      {"method": "gradcam++"}),
            "EigenCAM":  (eigencam_result,  explainer,      {"method": "eigencam"}),
            "XGradCAM":  (xgradcam_result,  explainer,      {"method": "xgradcam"}),
            "LIME":      (lime_result,      None, {}),
            # input_image is optional in IG's explain() so stability reruns
            # only need input_tensor — no extra kwargs required
            #"IG":        (ig_result,        ig_explainer,   {}),
        },
        input_tensor=p_img,
        input_image=img_np,
        stability_runs=10,
        noise_std=0.05,
    )

    cmp.report(results)
    cmp.plot(results, save_png=True, filename="comparison_test.png")

if __name__ == "__main__":
    test()