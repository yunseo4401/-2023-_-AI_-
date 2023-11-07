import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import get_scheduler

# 데이터 전처리 함수
def preprocess_function(examples):
    """
    주어진 데이터를 모델 입력 및 타겟 레이블로 전처리합니다.

    Args:
        examples (dict): 입력 데이터 및 결과 데이터를 포함하는 딕셔너리.
            - "content" (str): 입력 데이터.
            - "result" (str): 결과 데이터.

    Returns:
        dict: 모델의 입력 및 타겟 레이블을 포함하는 딕셔너리.
            - "input_ids" (list of int): 모델 입력의 토큰 ID 리스트.
            - "attention_mask" (list of int): 입력 데이터의 어텐션 마스크 리스트.
            - "labels" (list of int): 모델의 타겟 레이블로 사용될 토큰 ID 리스트.
    """
    model_inputs = tokenizer(
        examples["content"], max_length=max_input_length, truncation=True
    )
    # 타겟을 위한 토크나이저 설정
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["result"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터 로드
raw_data = pd.read_csv("eda완.csv")

# 데이터셋 분할
train_dataset = Dataset.from_pandas(raw_data[['content', 'result']][:int(len(raw_data)*0.8)])
val_dataset = Dataset.from_pandas(raw_data[['content', 'result']][int(len(raw_data)*0.8):int(len(raw_data)*0.9)])
test_dataset = Dataset.from_pandas(raw_data[['content', 'result']][int(len(raw_data)*0.9):])

# T5 모델 및 토크나이저 설정
checkpoint = "eenzeenee/t5-base-korean-summarization"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

max_input_length = 888 
max_target_length = 256
batch_size = 8

# 데이터셋 전처리 및 데이터 포맷 설정
train_dataset = train_dataset.map(
    preprocess_function, batched=True, batch_size=batch_size, remove_columns=["content", "result"]
)
train_dataset.set_format("torch")

val_dataset = val_dataset.map(
    preprocess_function, batched=True, batch_size=batch_size, remove_columns=["content", "result"]
)
val_dataset.set_format("torch")

# 데이터로더 및 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
features = [train_dataset[i] for i in range(2)]
data_collator(features)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

eval_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=data_collator,
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# 옵티마이저 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 가속기 설정
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# 러닝 레이트 스케줄러 설정
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 디바이스 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Early Stopping 관련 설정
early_stopping_patience = 3  # 몇 번 이상 검증 손실이 감소하지 않을 때 학습을 중단할 것인지 설정
best_val_loss = float("inf")  # 가장 낮은 검증 손실을 저장할 변수
early_stopping_counter = 0    # Early Stopping 카운터

progress_bar = tqdm(range(num_training_steps), desc=f"Epoch 0")

model.train()
for epoch in range(num_train_epochs):
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()

        if step % 50 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

        progress_bar.update(1)

    average_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch: {epoch}, Average Loss: {average_loss}")

    # 검증 데이터로 평가하기
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

    average_val_loss = val_loss / len(eval_dataloader)
    print(f"Epoch: {epoch}, Validation Loss: {average_val_loss}")
    model.train()

    # Early Stopping 검사
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early Stopping at Epoch {epoch}. Best Validation Loss: {best_val_loss}")
            break

    progress_bar.set_description(f"Epoch {epoch + 1}")

# 모델 저장
torch.save(model.state_dict(), 'model_GPU_EDA완.pth')
