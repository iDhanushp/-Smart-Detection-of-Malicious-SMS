"""Quick verification: is the loaded tflite actually the new trained model?"""
import numpy as np, json
import tensorflow as tf

MODEL_PATH = r'..\sms_fraud_detectore_app\assets\advanced_fraud_detector.tflite'
CONFIG_PATH = r'..\sms_fraud_detectore_app\assets\behavioral_model_config.json'

model = tf.lite.Interpreter(MODEL_PATH)
model.allocate_tensors()
inp = model.get_input_details()
out = model.get_output_details()
print(f'Input  shape : {inp[0]["shape"]}')
print(f'Output shape : {out[0]["shape"]}')

with open(CONFIG_PATH) as f:
    cfg = json.load(f)
mean  = np.array(cfg['scaler_mean'],  dtype=np.float32)
scale = np.array(cfg['scaler_scale'], dtype=np.float32)
print(f'Scaler non-zero entries : {(mean != 0).sum()} / {len(mean)}')
print(f'mean[:5]  : {mean[:5].tolist()}')

CLASSES = ['LEGIT', 'SPAM', 'FRAUD']

def predict(feat):
    f = (np.array(feat, dtype=np.float32) - mean) / np.where(scale == 0, 1, scale)
    model.set_tensor(inp[0]['index'], f.reshape(1, 30))
    model.invoke()
    p = model.get_tensor(out[0]['index'])[0]
    return p, CLASSES[int(np.argmax(p))]

print('\n--- Sanity tests ---')

# 1. Pure zeros (no features at all)
p, label = predict(np.zeros(30))
print(f'All-zero features       : L={p[0]:.3f} S={p[1]:.3f} F={p[2]:.3f}  => {label}')

# 2. Strong fraud signal: phone sender + money + data request + url + high fraudRisk
feat_fraud = np.zeros(30, dtype=np.float32)
feat_fraud[0]  = 0.5   # urgency
feat_fraud[2]  = 0.5   # fear
feat_fraud[4]  = 1.0   # money rewards
feat_fraud[8]  = 1.0   # data request
feat_fraud[22] = 1.0   # has_url
feat_fraud[24] = 1.0   # is_phone sender
feat_fraud[27] = 1.0   # fraudRisk = 1
feat_fraud[28] = 0.0   # spamRisk
feat_fraud[29] = 0.0   # legitScore
p, label = predict(feat_fraud)
print(f'Strong fraud signals    : L={p[0]:.3f} S={p[1]:.3f} F={p[2]:.3f}  => {label}')

# 3. Pure legit signal: DLT sender (isService=1), high legitScore, OTP-like
feat_legit = np.zeros(30, dtype=np.float32)
feat_legit[25] = 1.0   # is_service sender
feat_legit[29] = 1.0   # legitScore
feat_legit[27] = 0.0   # fraudRisk
feat_legit[28] = 0.0   # spamRisk
p, label = predict(feat_legit)
print(f'Pure legit DLT          : L={p[0]:.3f} S={p[1]:.3f} F={p[2]:.3f}  => {label}')

# 4. Spam signal: URL + reward keywords
feat_spam = np.zeros(30, dtype=np.float32)
feat_spam[5]   = 0.8   # reward keywords
feat_spam[22]  = 1.0   # has_url
feat_spam[25]  = 1.0   # is_service sender
feat_spam[28]  = 1.0   # spamRisk
p, label = predict(feat_spam)
print(f'Spam (reward+url)       : L={p[0]:.3f} S={p[1]:.3f} F={p[2]:.3f}  => {label}')

print('\n--- Old model check (old model trained on random noise should give ~0.33 each) ---')
# If probabilities cluster near 0.33 for all inputs, it's the old synthetic-noise model
results = []
for _ in range(10):
    f = np.random.rand(30).astype(np.float32)
    p, _ = predict(f)
    results.append(p)
arr = np.array(results)
print(f'Random input mean probs : L={arr[:,0].mean():.3f} S={arr[:,1].mean():.3f} F={arr[:,2].mean():.3f}')
print(f'Std dev                 : L={arr[:,0].std():.3f} S={arr[:,1].std():.3f} F={arr[:,2].std():.3f}')
print('(Old noise model would show all near 0.33 with low std. New model shows high variance.)')
