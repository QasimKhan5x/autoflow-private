from __future__ import absolute_import
from numpy import source
import torch
import logging
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
              RobertaConfig, RobertaModel, RobertaTokenizer)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("device: %s",device)

# build model
tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')
config = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
# import！！！you must set is_decoder as True for generation
config.is_decoder = True
encoder = RobertaModel.from_pretrained('microsoft/unixcoder-base',config=config) 

zmodel = Seq2Seq(encoder=encoder,decoder=encoder,config=config,
              beam_size=10 ,max_length=128,
              sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)

print('zmodel Seq2Seq Loaded')


zmodel = zmodel.module if hasattr(zmodel, 'module') else zmodel  
zmodel.load_state_dict(torch.load('./python/pytorch_model.bin', map_location=device))

print('Weights Loaded')

zmodel.to(device)


def inference(context):
    source_tokens = tokenizer.tokenize(context)[:512-5]
    source_tokens = [tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token,"<mask0>"]+source_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens) 
    padding_length = 512 - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length



    all_source_ids = torch.tensor([source_ids], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=48)
    p=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids = batch[0]
        source_ids.to(device)                  
        with torch.no_grad():
            preds = zmodel(source_ids)   
            # convert ids to text
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)
    return p
    
