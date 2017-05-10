# Seq2Seq Chatbot with Tensorflow

by Park Ji Ho 
from Hong Kong University of Science and Technology(HKUST) Human Language Technology Center

This is a seq2seq chatbot from the Neural Conversation Model(Vinyals, O., and Le, Q. 2015. A neural conversational model. Proceedings of the International Conference on Machine Learning, Deep Learning Workshop) implemented with Tensorflow 

## Scripts 
1. `seq2seq_wrapper.py`: seq2seq model
2. `train.py`: functions for training the module
3. `predict.py`: decoding part (beam-search)
4. `chat.py`: module for load the pre-trained module for chatting.
5. `data_utils.py`: utils functions (batch generation)

## Training
Look at `twitter-chatbot.py`. The checkpoints and summary will be saved in `logs`

## Decoding
Look at `predict.py`. Now uses random beam search.

## Chatting
Pretrained model can be used to chat.
See `chat.py`. We incorporated this module with Flask in another project. (Flask code is not here)

### Examples

| Question					| Chatbot Reply					|
| ------------- | ------------- |
|hahaha :p i'm okay thanks :) you ? |i am good , i'm just tired . how are you ? |
|im the same person all the time :-) |yeah true ! ! i bet u could talk to me haha |
|yess , girl his dusty ' ass sent me a friend request on fb . he swear he so funny ! | omg i see i am tryna get him out ! ! |
| please do me a favor . get on twitter more often . creep me sometimes , talk like before . thank you . |uh , i just don't want you to get my ass up  | 
| understandable , he's in my prayers . well i won't keep you any longer ;-) i hadn't spoken to you forever i n    eeded to reach out !  | i know , i think i will be better soon . don't know how to get me to work . |
| crazy . so how is your summer going ? still bored at home with zero funds ?  |by the point , and i don't know what doing it anymore . but then ... |
|why you haven't texted me ? |because you don't want me |
| why does boredom encourage hunger ? i must admit , i am peckish . | you're welcome . and the difference is my family . but thank you .  |
| work is hectic and endless mood swings  | what is it ? ? |
|  my schedule is very hectic right now but you definitly got my attention with the compliments . compliments wi    ll get you far :)  | indeed ! ! ! ill get through it ! ! |
|only an insurance list lol doesn't really matter , seeing as it's only a week apart in delivery dates as we w    ere in ny then delayed  |because i don't wanna have one in the morning :(  |
|  * looks at her and sighs * | * smiles and takes her hand * |
| haha yaa ! i want more but my mom wont let me :/  | go to one of your friends then ? |
|  * rolls her eyes * alright , s . | shame |
| do tell . if it's a simple plan , i already heard ' em ... and love ' em | i don't even know how to tell . * pokes her lips * i don't know what i do . |
| sure . it'll take a bit of time since i'm starting white next week . any special ability ? you'll have to rem    ind me later too ... | i will see you drive through <name> . i'm not going to move with him . i'd love to have to maybe  |
| * laughs * my adorable wife * tugs at your lower lip * | * blushes * you need to go ! ! ! |
|  sounds funn i hope yu get it (:  | lol well i'm sure i was gonna go for it  |
| i got special powers | why not ? |
|  & & i still waiting to see your hair . did it get better ? | lol you're mad ! |
| yeah me to . btw , how long do u can stay ? and are you busy tommorow ? | might probably be at work :-( |
| wack ... lol | lol i always love it |
| lmao ... um not gonna front , her single automatic is catchy , i like it but the 2nd season is going to be ab    t what ?  | true . i think he will already have money in the second ! ?  |
| boy stop lol  | oh wait ! |
| check that out - explicit but hilarious . | what do you mean ? ! ?  |
|  i hug . " i am happy to see you . " i hug more . | thank you . * hugs you * |
| follow back so i can dm u my number  | what ? ?  |
| sets cup down too , standing up behind you and wrapping arms around you -  | smiles and holds you tight - i missed you |
| that's so amazing ! ! :) | yes it is ! and i'm going to kill it ! |
| that is so cool ! ! i live in australia (: | lovely ! :) | 

## Datasets
Each dataset has separate `data.py` for preprocessing

- Twitter Dialogue Pairs from [julianser](https://github.com/julianser/hed-dlg-truncated)
Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models. Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau. 2016c. AAAI. http://arxiv.org/abs/1507.04808.

+ you can add other corpus you have easily. for reference, see  `twitter-chatbot.py` and preprocessing `datasets/tweets/data.py`

## Requirements
- tensorflow 1.0.0 (higher version might have error due to tf.legacy_seq2seq)
- python 3.4
- numpy
- pandas (for loading data)
- tqdm (for preprocessing)

## Remarks
- Takes pretty long time to train (1M training pairs around 15+ hours with GPU). Decoding takes 4~5 secs in GPU, but over 60 secs in slow CPU. 
- Improvements on decoding needed (eg. rescoring, diversity promoting) 

## Credits
- Forked from this work [**Practical seq2seq**](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/), for more details.
