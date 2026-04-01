import os
import time
import wave
import json
import numpy as np
import pyaudio
import re
import platform
import requests
from collections import deque
from vosk import Model, KaldiRecognizer
from faster_whisper import WhisperModel

import ai

# config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
BUFFER_SEC = 2 

# colors
c_red = "\033[91m"
c_orange = "\033[93m"
c_blue = "\033[94m"
c_green = "\033[92m"
c_cyan = "\033[96m"
c_reset = "\033[0m"

def check_models():
    # check models
    try:
        res = requests.get("http://localhost:11434/api/tags")
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            if "ministral-3:3b" in models:
                print(f"{c_green}[status] models loaded{c_reset}")
            else:
                print(f"{c_orange}[status] models loading...{c_reset}")
        else:
            print(f"{c_red}[status] ollama error{c_reset}")
    except:
        print(f"{c_red}[status] ollama offline{c_reset}")

def record_until_silence(stream, pre_buffer):
    # init
    frames = list(pre_buffer)
    silence_limit = int((RATE / CHUNK) * 2.0)
    max_record = int((RATE / CHUNK) * 15.0)
    timeout_limit = int((RATE / CHUNK) * 5.0)
    
    silence_count = 0
    timeout_count = 0
    has_spoken = False

    # record
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        pcm_np = np.frombuffer(data, dtype=np.int16)
        volume = np.max(np.abs(pcm_np))
        
        if volume > 200:
            has_spoken = True
            silence_count = 0
        else:
            if has_spoken:
                silence_count += 1
            else:
                timeout_count += 1
                
        if has_spoken and silence_count > silence_limit:
            break
            
        if not has_spoken and timeout_count > timeout_limit:
            break
            
        if len(frames) > max_record:
            break

    # save
    temp_wav = "temp_capture.wav"
    wf = wave.open(temp_wav, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_wav

def transcribe_audio(file_path, stt_model):
    # transcribe
    try:
        segments, _ = stt_model.transcribe(file_path, language="fr")
        text = " ".join([segment.text for segment in segments])
        
        os.remove(file_path)
        return text.strip()
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return ""

def main():
    # init
    print(f"{c_red}[init] starting systems...{c_reset}")
    
    ai.verify_and_pull_models()
    check_models()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    model_path = "model-fr" 
    try:
        vosk_model = Model(model_path)
    except Exception:
        print(f"{c_red}vosk missing{c_reset}")
        return
        
    rec = KaldiRecognizer(vosk_model, RATE, '["maestro", "[unk]"]')
    whisper_model = WhisperModel("medium", device="auto", compute_type="int8")

    buffer_len = int((RATE / CHUNK) * BUFFER_SEC)
    ring_buffer = deque(maxlen=buffer_len)

    follow_up_mode = False
    follow_up_timer = 0
    
    system_msg = "tu es maestro, un assistant vocal intelligent. tu réponds toujours de manière directe, concise et naturelle."
    conversation_history = [{"role": "system", "content": system_msg}]

    print(f"{c_cyan}[state] waiting wake word{c_reset}")

    # loop
    while True:
        pcm = stream.read(CHUNK, exception_on_overflow=False)
        pcm_np = np.frombuffer(pcm, dtype=np.int16)
        
        ring_buffer.append(pcm)
        trigger = False

        if not follow_up_mode:
            # check wake
            if rec.AcceptWaveform(pcm):
                result = json.loads(rec.Result())
                if "maestro" in result.get("text", ""):
                    trigger = True
        else:
            # check timeout
            if time.time() - follow_up_timer > 8:
                follow_up_mode = False
                print(f"\n{c_cyan}[state] waiting wake word{c_reset}")
            else:
                # check vol
                if np.max(np.abs(pcm_np)) > 400: 
                    trigger = True

        if trigger:
            print(f"\n{c_green}[state] listening...{c_reset}")
            
            # record
            audio_path = record_until_silence(stream, ring_buffer)
            
            print(f"{c_red}[state] transcribing...{c_reset}")
            
            # stt
            user_text = transcribe_audio(audio_path, whisper_model)
            
            if user_text:
                print(f"user: {user_text}")
                
                conversation_history.append({"role": "user", "content": user_text})
                print(f"{c_orange}[state] thinking...{c_reset}")
                
                start_t = time.time()
                
                # llm
                model_name, response_stream = ai.route_llm_request(conversation_history)
                
                print(f"{c_blue}[state] responding [{model_name}]: ", end="", flush=True)
                full_reply = ""
                
                # read stream
                for chunk in response_stream:
                    print(f"{c_blue}{chunk}{c_reset}", end="", flush=True)
                    full_reply += chunk
                
                elapsed = time.time() - start_t
                print(f"\n[time: {elapsed:.2f}s]")
                
                conversation_history.append({"role": "assistant", "content": full_reply})
                
                # state update
                follow_up_mode = True
                follow_up_timer = time.time()
                print(f"{c_green}[state] follow up listening...{c_reset}")
            else:
                print("no text")
                if follow_up_mode:
                    # reset timer
                    follow_up_timer = time.time()
                    print(f"{c_green}[state] follow up listening...{c_reset}")
                else:
                    print(f"{c_cyan}[state] waiting wake word{c_reset}")
            
            # reset
            ring_buffer.clear()
            
            # clean
            while stream.get_read_available() > 0:
                stream.read(stream.get_read_available(), exception_on_overflow=False)

if __name__ == "__main__":
    # start
    main()