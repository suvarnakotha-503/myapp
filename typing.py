{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1cb37e90",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "from dataclasses import dataclass, asdict\n",
    "import numpy as np\n",
    "from pynput import keyboard\n",
    "from sklearn.svm import OneClassSVM\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "# ------------------ Keystroke Event ------------------\n",
    "@dataclass\n",
    "class KeystrokeEvent:\n",
    "    key: str\n",
    "    t_down: float\n",
    "    t_up: float\n",
    "\n",
    "# ------------------ Typing Collector ------------------\n",
    "class TypingCollector:\n",
    "    def __init__(self):\n",
    "        self._pressed = {}\n",
    "        self._events = []\n",
    "        self._start = None\n",
    "        self._end = None\n",
    "\n",
    "    def _key_to_str(self, k):\n",
    "        try:\n",
    "            if hasattr(k, 'char') and k.char: return k.char\n",
    "            if str(k) == 'Key.space': return ' '\n",
    "            if str(k) == 'Key.backspace': return '<BKSP>'\n",
    "            if str(k) == 'Key.enter': return '<ENTER>'\n",
    "        except: return None\n",
    "        return None\n",
    "\n",
    "    def _on_press(self, k):\n",
    "        s = self._key_to_str(k)\n",
    "        if self._start is None: self._start = time.perf_counter()\n",
    "        if s is None: return\n",
    "        if s == '<ENTER>':\n",
    "            self._end = time.perf_counter()\n",
    "            return False\n",
    "        self._pressed[id(k)] = time.perf_counter()\n",
    "\n",
    "    def _on_release(self, k):\n",
    "        s = self._key_to_str(k)\n",
    "        if s is None: return\n",
    "        now = time.perf_counter()\n",
    "        if id(k) in self._pressed:\n",
    "            self._events.append(KeystrokeEvent(s, self._pressed[id(k)], now))\n",
    "            del self._pressed[id(k)]\n",
    "\n",
    "    def collect(self):\n",
    "        print(\"Type ANYTHING you want, then press Enter:\\n\")\n",
    "        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:\n",
    "            listener.join()\n",
    "        return {\n",
    "            \"events\": [asdict(e) for e in self._events],\n",
    "            \"duration\": (self._end - self._start) if self._end else None\n",
    "        }\n",
    "\n",
    "# ------------------ Feature Extraction ------------------\n",
    "def extract_features(sample):\n",
    "    ev = [KeystrokeEvent(**e) for e in sample[\"events\"]]\n",
    "    if not ev:\n",
    "        return np.zeros(15), {\"wpm\":0,\"duration\":0}\n",
    "    ev.sort(key=lambda x: x.t_down)\n",
    "\n",
    "    dwell = np.array([e.t_up - e.t_down for e in ev])\n",
    "    press = np.array([e.t_down for e in ev])\n",
    "    flight = np.diff(press) if len(press) >= 2 else np.array([0.0])\n",
    "    duration = max([e.t_up for e in ev]) - min([e.t_down for e in ev])\n",
    "\n",
    "    cps = len(ev)/duration if duration > 0 else 0\n",
    "    wpm = cps*60/5\n",
    "\n",
    "    def stats(arr):\n",
    "        return [np.mean(arr), np.std(arr), np.percentile(arr,25),\n",
    "                np.percentile(arr,50), np.percentile(arr,75)]\n",
    "\n",
    "    feat = []\n",
    "    feat += stats(dwell)\n",
    "    feat += stats(flight)\n",
    "    feat += [wpm, duration]\n",
    "    return np.array(feat), {\"wpm\":wpm,\"duration\":duration}\n",
    "\n",
    "# ------------------ Train + Compare ------------------\n",
    "def train_profile(user_features):\n",
    "    pipe = Pipeline([\n",
    "        (\"scaler\", StandardScaler()),\n",
    "        (\"ocsvm\", OneClassSVM(kernel=\"rbf\", nu=0.1, gamma=\"scale\"))\n",
    "    ])\n",
    "    pipe.fit(user_features)\n",
    "    return pipe\n",
    "\n",
    "# ------------------ Enrollment & Authentication ------------------\n",
    "profile_model = None\n",
    "\n",
    "def enroll_user(n_samples=3):\n",
    "    global profile_model\n",
    "    feats = []\n",
    "    for i in range(n_samples):\n",
    "        print(f\"\\n[Enrollment Sample {i+1}/{n_samples}]\")\n",
    "        sample = TypingCollector().collect()\n",
    "        feat, meta = extract_features(sample)\n",
    "        feats.append(feat)\n",
    "        print(f\"  -> WPM={meta['wpm']:.1f}, Duration={meta['duration']:.2f}s\")\n",
    "    X = np.vstack(feats)\n",
    "    profile_model = train_profile(X)\n",
    "    print(\"\\n✅ User profile enrolled.\")\n",
    "\n",
    "def authenticate_user():\n",
    "    global profile_model\n",
    "    if profile_model is None:\n",
    "        print(\"❌ No profile enrolled yet.\")\n",
    "        return\n",
    "    sample = TypingCollector().collect()\n",
    "    feat, meta = extract_features(sample)\n",
    "    score = profile_model.decision_function([feat])[0]\n",
    "    print(f\"\\nScore={score:.3f}  |  WPM={meta['wpm']:.1f}, Duration={meta['duration']:.2f}s\")\n",
    "    if score > 0:\n",
    "        print(\"✅ VALID USER (Access Granted)\")\n",
    "    else:\n",
    "        print(\"⛔ INVALID USER (Access Denied)\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "5199c03f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Enrollment Sample 1/3]\n",
      "Type ANYTHING you want, then press Enter:\n",
      "\n",
      "  -> WPM=0.0, Duration=0.00s\n",
      "\n",
      "[Enrollment Sample 2/3]\n",
      "Type ANYTHING you want, then press Enter:\n",
      "\n",
      "  -> WPM=0.0, Duration=0.00s\n",
      "\n",
      "[Enrollment Sample 3/3]\n",
      "Type ANYTHING you want, then press Enter:\n",
      "\n",
      "  -> WPM=149.8, Duration=0.08s\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 15 and the array at index 2 has size 12",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[23], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m enroll_user(n_samples\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m)\n",
      "Cell \u001b[1;32mIn[16], line 105\u001b[0m, in \u001b[0;36menroll_user\u001b[1;34m(n_samples)\u001b[0m\n\u001b[0;32m    103\u001b[0m     feats\u001b[38;5;241m.\u001b[39mappend(feat)\n\u001b[0;32m    104\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m  -> WPM=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mmeta[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mwpm\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;132;01m:\u001b[39;00m\u001b[38;5;124m.1f\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m, Duration=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mmeta[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mduration\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;132;01m:\u001b[39;00m\u001b[38;5;124m.2f\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124ms\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m--> 105\u001b[0m X \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mvstack(feats)\n\u001b[0;32m    106\u001b[0m profile_model \u001b[38;5;241m=\u001b[39m train_profile(X)\n\u001b[0;32m    107\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m✅ User profile enrolled.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m<__array_function__ internals>:200\u001b[0m, in \u001b[0;36mvstack\u001b[1;34m(*args, **kwargs)\u001b[0m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\numpy\\core\\shape_base.py:296\u001b[0m, in \u001b[0;36mvstack\u001b[1;34m(tup, dtype, casting)\u001b[0m\n\u001b[0;32m    294\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(arrs, \u001b[38;5;28mlist\u001b[39m):\n\u001b[0;32m    295\u001b[0m     arrs \u001b[38;5;241m=\u001b[39m [arrs]\n\u001b[1;32m--> 296\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m _nx\u001b[38;5;241m.\u001b[39mconcatenate(arrs, \u001b[38;5;241m0\u001b[39m, dtype\u001b[38;5;241m=\u001b[39mdtype, casting\u001b[38;5;241m=\u001b[39mcasting)\n",
      "File \u001b[1;32m<__array_function__ internals>:200\u001b[0m, in \u001b[0;36mconcatenate\u001b[1;34m(*args, **kwargs)\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 15 and the array at index 2 has size 12"
     ]
    }
   ],
   "source": [
    "enroll_user(n_samples=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b5ed7123",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1614634978.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[19], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    hi hello good evening\u001b[0m\n\u001b[1;37m       ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "hi hello good evening\n",
    "sun rises in the east\n",
    "moon rises in the west\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "253b07eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Type ANYTHING you want, then press Enter:\n",
      "\n",
      "\n",
      "Score=-0.117  |  WPM=21.6, Duration=9.98s\n",
      "⛔ INVALID USER (Access Denied)\n"
     ]
    }
   ],
   "source": [
    "authenticate_user()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "819fe870",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'suvarna' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[22], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m suvarna\n",
      "\u001b[1;31mNameError\u001b[0m: name 'suvarna' is not defined"
     ]
    }
   ],
   "source": [
    "suvarna\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5c11fe2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b7448f4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
