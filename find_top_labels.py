from load_my import *
import torchmetrics
from tqdm import tqdm


def initialize_counter_dict():
    dict_ = dict()
    for actual_label in label_to_classname:
        dict_[actual_label] = dict()
        for predicted_label in label_to_classname:
            dict_[actual_label][predicted_label] = 0
    return dict_


seed_everything(hparams['seed'])

bs = 1
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=40, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model) # Original
# description_encodings = compute_description_encodings(model, True) # My addition

label_encodings = compute_label_encodings(model)


print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=200).to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=200).to(device)

clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=200).to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=200).to(device)

count_classes = initialize_counter_dict()


for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    top5_values, top5_indices = image_labels_similarity.topk(5, dim=1)

    actual_class_name = label_to_classname[labels.squeeze()]
    for predicted_index in top5_indices.squeeze():
        predicted_class_name = label_to_classname[predicted_index]
        count_classes[actual_class_name][predicted_class_name] +=1
    
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
#     top5_values, top5_indices = cumulative_tensor.topk(5, dim=1)

#     actual_class_name = label_to_classname[labels.squeeze()]
#     for predicted_index in top5_indices.squeeze():
#         predicted_class_name = label_to_classname[predicted_index]
#         count_classes[actual_class_name][predicted_class_name] +=1
    
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



for actual_keys in count_classes.keys():
    count_classes[actual_keys] = sorted(count_classes[actual_keys].items(), key=lambda x: x[1], reverse=True)

json_string = json.dumps(count_classes, indent=4)  # indent for pretty formatting

# Write JSON string to a text file
with open('top5.txt', 'w') as file:
    file.write(json_string)

