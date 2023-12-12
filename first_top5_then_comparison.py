from load import *
import torchmetrics
from tqdm import tqdm

def perform_original():
    print("Performing Original....")
    print("Encoding descriptions...")

    description_encodings = compute_description_encodings(model)

    label_encodings = compute_label_encodings(model)

    print("Evaluating...")
    lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=n_classes).to(device)

    clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=n_classes).to(device)

    for batch_number, batch in enumerate(tqdm(dataloader)):
        images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        image_labels_similarity = image_encodings @ label_encodings.T
        clip_predictions = image_labels_similarity.argmax(dim=1)
        
        
        clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
        clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
        
        
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        
        for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
            
            
            dot_product_matrix = image_encodings @ v.T
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
            
            
        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
            
        
        descr_predictions = cumulative_tensor.argmax(dim=1)
        
        
        lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
        
        

    print("\n")

    accuracy_logs = {}
    accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
    accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

    accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
    accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

    # print the dictionary
    print("\n")
    for key, value in accuracy_logs.items():
        print(key, value)


def perform_mine():
    print("\n\nPerforming Mine....")
    print("Encoding descriptions...")

    description_encodings = compute_description_encodings_mine_oxp(model)

    label_encodings = compute_label_encodings(model)

    print("Evaluating...")
    lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=n_classes).to(device)

    clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=n_classes).to(device)



    for batch_number, batch in enumerate(tqdm(dataloader)):
        images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)
        
        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        image_labels_similarity = image_encodings @ label_encodings.T
        _, topk_indices = image_labels_similarity.topk(5, dim=1)
        clip_predictions = image_labels_similarity.argmax(dim=1)
        
        
        clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
        clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)

        index_to_class = idx_to_label(dataset.class_to_idx)
        predictions = []

        for index, top_indices in enumerate(topk_indices):
            top_classes = list(map(lambda x: index_to_class[int(x)], top_indices.squeeze()))
            top_classes_description_encodings_list = []
            for top_class in top_classes:
                top_classes_description_encodings_list.append(description_encodings[top_class])

            top_classes_description_encodings = torch.stack(top_classes_description_encodings_list, dim = 1).squeeze()
            # print(f"top_classes_description_encodings.size(): {top_classes_description_encodings.size()}")
            dot_product_matrix = image_encodings[index] @ top_classes_description_encodings.T
            predictions.append(top_indices[dot_product_matrix.argmax(dim=0)])
        predictions = torch.stack(predictions, dim = 0)
        # print(f"predictions.size(): {predictions.size()}")
        # print(f"Labels size: ", labels.size())
        lang_acc = lang_accuracy_metric(predictions, labels)
        
        

    print("\n")

    accuracy_logs = {}
    accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
    # accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

    accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
    accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

    # print the dictionary
    print("\n")
    for key, value in accuracy_logs.items():
        print(key, value)

seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)
# perform_original() # Perform original process

perform_mine()