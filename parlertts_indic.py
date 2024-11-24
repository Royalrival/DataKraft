from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"


model = ParlerTTSForConditionalGeneration.from_pretrained("En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

# Input text in Devanagari
prompt_in_devanagari = "भारत में कई प्रकार की भाषाएँ बोली जाती हैं, जिनमें से हिंदी सबसे प्रमुख है। यह भाषा लोगों को एकजुट करने में महत्वपूर्ण भूमिका निभाती है।"
description = "'Male Hindi Speaker, in Hindi delivers his words slowly and clearly, in a very calm sounding environment with clear audio quality. He speaks fast.'"

# Transliterate the Devanagari prompt to Latin script (ABCD)
prompt_transliterated = transliterate(prompt_in_devanagari, DEVANAGARI, ITRANS)
#prompt_transliterated = "Bhārat mein vividhatāon kā ek adbhut saṅgam hai. Yahān ke vividh pradeśon, bhāṣāon, sanskṛtiyon, aur paramparāon mein bemisāl samṛddhi hai. Chāhe uttara ho yā dakṣiṇa, paścima ho yā pūrba, har kṣetra kī apnī ek khās pahchān hai. Rājasthān kī registānī havā, Kashmīr kī barfilī vādīyān, Keral ke hare-bhare baikwāṭars, aur Baṅgāl kī sānskṛtik dhārā, sabhī kuchh milkar is viśāl deś kī advitīyatā ko pradarśit karte hain. Bhāratīya samāj mein parivār, mitratā aur sahanshīltā kā mahatva atyadhik hai, aur yah hamāre jīvan ke mūlbhūt tatvon mein se ek hai."
print("Prompt: ", prompt_in_devanagari)
print("Prompt Transliterated: ", prompt_transliterated)
# Tokenize inputs
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt_transliterated, return_tensors="pt").input_ids.to(device)

# Generate audio
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

from IPython.display import Audio
Audio(audio_arr, rate=model.config.sampling_rate)

import soundfile as sf
import numpy as np

# Convert audio array to float32
audio_arr_float32 = audio_arr.astype(np.float32)

# Specify the file path where you want to save the audio
output_path = 'generated_audio.wav'

# Write the converted audio array to a WAV file
sf.write(output_path, audio_arr_float32, model.config.sampling_rate)

print(f"Audio saved to {output_path}")