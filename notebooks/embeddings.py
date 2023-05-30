#!/usr/bin/env python
# coding: utf-8

# ## How embeddings extracted from text in Kaggle evaluation kernel

# In[1]:


import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('/home/toomuch/kaggle-diffusion/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

comp_path = Path('/data/kaggle/image2prompt')


# In[2]:


prompts = pd.read_csv(comp_path / 'prompts.csv', index_col='imgId')
prompts.head(7)


# In[3]:


sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
sample_submission.head()


# In[4]:


st_model = SentenceTransformer('/home/toomuch/kaggle-diffusion/all-MiniLM-L6-v2')
prompt_embeddings = st_model.encode(prompts['prompt']).flatten()


# In[5]:


assert np.all(np.isclose(sample_submission['val'].values, prompt_embeddings, atol=1e-07))


# ## Extract biased texts from diffusion-db
# Pipeline:
# 1. Extract texts & save them somewhere
# 1. Extract CLIP and MiniLM embeddings from them
# 1. Try to approximate MiniLM embeddings using CLIP and MLP

# In[6]:


import pandas as pd

df = pd.read_parquet('./metadata-large.parquet')
len(list(df['prompt'].unique()))


# In[7]:


df.head()


# In[8]:


import hashlib
from tqdm import tqdm


# In[9]:


df['id'] = [hashlib.md5(el.encode('utf-8')).hexdigest()[:8] for el in tqdm(df['prompt'])]

print(len(df['id'].unique()) / len(df['prompt'].unique()))


# In[10]:


from transformers import AutoModel, AutoTokenizer
# import torch 
openclip_model = AutoModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to('cuda:2')
openclip_tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

# class Model(torch.nn.Module):
#     def __init__(self)


# In[11]:


openclip_tokenizer(
    "whereas it goes",
    add_special_tokens=True,
    max_length=77,
    padding="max_length",
    return_token_type_ids=True,
    truncation=True,
)


# In[15]:


# df['prompt'].drop_duplicates().hist()
df['prompt'].apply(lambda x: len(x.split(' '))).quantile(q=0.985)


# In[16]:


import torch
import clip


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        assert len(df.columns) == 2
        self.pairs = df.values

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt_id, prompt = self.pairs[idx]
        while True:
            try:
                prompt_tensor_clip = clip.tokenize([prompt])
                break
            except RuntimeError:
                prompt = " ".join(prompt.split(" ")[:-1])

        return prompt_id, prompt, prompt_tensor_clip


class OpenClipDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        assert len(df.columns) == 2
        self.pairs = df.values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt_id, prompt = self.pairs[idx]
        prompt_tensor_clip = self.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )['input_ids']

        prompt_tensor_clip = torch.tensor(prompt_tensor_clip, dtype=torch.long)

        return prompt_id, prompt, prompt_tensor_clip


dataloader = torch.utils.data.DataLoader(
    # dataset=ClipDataset(df[["id", "prompt"]].drop_duplicates(subset=["id"])),
    dataset=OpenClipDataset(df[["id", "prompt"]].drop_duplicates(subset=["id"]), openclip_tokenizer),
    # batch_size=6144,
    batch_size=4096,
    num_workers=0,
    shuffle=False,
)


# In[17]:


openclip_model.text_projection = openclip_model.text_projection.to('cuda:2')
openclip_model.text_model = openclip_model.text_model.to('cuda:2')
st_model = st_model.to('cuda:2')


# In[18]:


# st_model.device


# In[19]:


# import os 
# os.environ['OMP_NUM_THREADS'] = "1"


# In[20]:


# openclip_model.encode_text


# In[21]:


dump = dict()
import time


torch.cuda.empty_cache()
for batch in tqdm(dataloader):
    prompt_ids, prompts, prompt_tensors_clip = batch
    with torch.no_grad():
        # print(type(st_model.encode(prompts)))
        # raise KeyboardInterrupt
        # print('st model')
        st_embeddings = st_model.encode(prompts).tolist()
        # time.sleep(3)
        # clip_embeddings = (
        #     clip_model.encode_text(prompt_tensors_clip.squeeze().to(device))
        #     .detach()
        #     .cpu()
        #     .numpy()
        #     .tolist()
        # )
        # print(prompt_tensors_clip)
        # print('to cuda')
        prompt_tensors_clip = prompt_tensors_clip.to('cuda:2')
        # time.sleep(3)
        # print('openclip text model')
        clip_embeddings = openclip_model.text_model(prompt_tensors_clip)['pooler_output']
        # time.sleep(3)
        # print('openclip text proj')
        clip_embeddings = openclip_model.text_projection(clip_embeddings)
        # time.sleep(3)
        # print(clip_embeddings.device)
        clip_embeddings = clip_embeddings.detach().cpu().numpy().tolist()
    # print(st_embeddings)
    # print(clip_embeddings)
    for _prompt_id, _st_emb, _clip_emb, _prompt in zip(prompt_ids, st_embeddings, clip_embeddings, prompts):
        dump[_prompt_id] = {'prompt': _prompt, 'MiniLM-emb': _st_emb, 'CLIP-emb': _clip_emb}
    # raise KeyboardInterrupt


# In[29]:


pd.DataFrame.from_dict(data=dump, orient='index').reset_index(drop=False).to_parquet('./a.parquet', index=False)


# In[ ]:


raise KeyboardInterrupt


# In[99]:


# df[['id', 'prompt']].loc(lambda x: len(x.split(' ')))


# In[100]:


# df[['id', 'prompt']].drop_duplicates(subset=['id'])


# In[115]:


# from transformers import AutoModel, AutoTokenizer

# openclip_model = AutoModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
# openclip_tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')


# In[113]:


# openclip_model.text_model


# In[110]:


#  'text_embed_dim',
#  'text_model',
#  'text_projection',
# dir(model)


# In[101]:


# # next(iter(dataloader))
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[96]:


device = "cuda:2" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("RN50", device="cpu", jit=False)
clip_model, preprocess = clip.load("ViT-L/14@336px", device="cpu", jit=False)
clip_model.eval()
clip_model = clip_model.to(device)

st_model = SentenceTransformer('/home/toomuch/kaggle-diffusion/all-MiniLM-L6-v2')
st_model = st_model.to(device)


# In[102]:


dump = dict()


torch.cuda.empty_cache()
for batch in tqdm(dataloader):
    prompt_ids, prompts, prompt_tensors_clip = batch
    with torch.no_grad():
        # print(type(st_model.encode(prompts)))
        # raise KeyboardInterrupt
        st_embeddings = st_model.encode(prompts).tolist()
        clip_embeddings = (
            clip_model.encode_text(prompt_tensors_clip.squeeze().to(device))
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
    # print(st_embeddings)
    # print(clip_embeddings)
    for _prompt_id, _st_emb, _clip_emb, _prompt in zip(prompt_ids, st_embeddings, clip_embeddings, prompts):
        dump[_prompt_id] = {'prompt': _prompt, 'MiniLM-emb': _st_emb, 'CLIP-emb': _clip_emb}
    # raise KeyboardInterrupt


# In[56]:


dump


# In[ ]:




