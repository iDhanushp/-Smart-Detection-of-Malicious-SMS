"""
Multilingual SMS Fraud Detection Training
Supports multiple languages with language-specific models and detection
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import os
from typing import List, Dict, Tuple, Optional
import re
from langdetect import detect, DetectorFactory
import pickle

# Set seed for consistent language detection
DetectorFactory.seed = 0

class MultilingualFraudDetector:
    """
    Multilingual SMS fraud detection system
    """
    
    def __init__(self):
        self.languages = ['en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ja', 'ko', 'ru']
        self.models = {}
        self.vectorizers = {}
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian'
        }
        
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the SMS text
        """
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            if len(cleaned_text) < 10:
                return 'en'  # Default to English for very short texts
                
            detected_lang = detect(cleaned_text)
            
            # Map similar languages
            if detected_lang in ['zh-cn', 'zh-tw']:
                detected_lang = 'zh'
            elif detected_lang in ['ja', 'jp']:
                detected_lang = 'ja'
            elif detected_lang in ['ko', 'kr']:
                detected_lang = 'ko'
                
            # Return detected language if supported, otherwise default to English
            return detected_lang if detected_lang in self.languages else 'en'
            
        except Exception as e:
            print(f"Language detection failed: {e}")
            return 'en'  # Default to English
    
    def _clean_text_for_detection(self, text: str) -> str:
        """
        Clean text for language detection
        """
        # Remove URLs, numbers, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_multilingual_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data for multiple languages
        """
        print("Preparing multilingual training data...")
        
        # Load the main dataset
        df = pd.read_csv(data_path)
        
        # Language-specific datasets
        language_data = {}
        
        for lang in self.languages:
            print(f"Processing {self.language_names[lang]} data...")
            
            if lang == 'en':
                # Use existing English data
                language_data[lang] = df.copy()
            else:
                # For other languages, we would need language-specific datasets
                # For now, create synthetic data or use translation
                language_data[lang] = self._create_synthetic_data(df, lang)
        
        return language_data
    
    def _create_synthetic_data(self, base_df: pd.DataFrame, language: str) -> pd.DataFrame:
        """
        Create synthetic data for languages without specific datasets
        This is a placeholder - in production, you'd use real translated data
        """
        # Create a smaller synthetic dataset for demonstration
        synthetic_data = []
        
        # Fraudulent messages in different languages
        fraud_messages = {
            'hi': [
                'आपका बैंक खाता निलंबित हो गया है। पुनर्सक्रियण के लिए क्लिक करें',
                'आपने 50000 रुपये जीते हैं। अभी क्लेम करें',
                'आपका Aadhaar नंबर ब्लॉक हो गया है। तुरंत वेरिफाई करें',
                'आपका पैन कार्ड एक्सपायर हो गया है। अपडेट करें',
            ],
            'es': [
                'Su cuenta bancaria ha sido suspendida. Haga clic para reactivar',
                'Ha ganado $1000. Reclame ahora',
                'Su tarjeta de crédito ha sido bloqueada. Verifique inmediatamente',
                'Su cuenta necesita verificación urgente',
            ],
            'fr': [
                'Votre compte bancaire a été suspendu. Cliquez pour réactiver',
                'Vous avez gagné 1000€. Réclamez maintenant',
                'Votre carte de crédit a été bloquée. Vérifiez immédiatement',
                'Votre compte nécessite une vérification urgente',
            ],
            'de': [
                'Ihr Bankkonto wurde gesperrt. Klicken Sie zur Reaktivierung',
                'Sie haben 1000€ gewonnen. Jetzt einfordern',
                'Ihre Kreditkarte wurde gesperrt. Sofort überprüfen',
                'Ihr Konto benötigt dringende Überprüfung',
            ],
            'zh': [
                '您的银行账户已被暂停。点击重新激活',
                '您赢得了1000元。立即领取',
                '您的信用卡已被冻结。立即验证',
                '您的账户需要紧急验证',
            ],
            'ar': [
                'تم تعليق حسابك المصرفي. انقر لإعادة التفعيل',
                'لقد ربحت 1000 دولار. اطلب الآن',
                'تم حظر بطاقة الائتمان الخاصة بك. تحقق فوراً',
                'حسابك يحتاج إلى تحقق عاجل',
            ],
            'ja': [
                'あなたの銀行口座が停止されました。再開するにはクリックしてください',
                '1000円を獲得しました。今すぐ請求してください',
                'あなたのクレジットカードがブロックされました。すぐに確認してください',
                'あなたのアカウントは緊急確認が必要です',
            ],
            'ko': [
                '귀하의 은행 계좌가 일시정지되었습니다. 재개하려면 클릭하세요',
                '1000원을 획득했습니다. 지금 청구하세요',
                '귀하의 신용카드가 차단되었습니다. 즉시 확인하세요',
                '귀하의 계정은 긴급 확인이 필요합니다',
            ],
            'ru': [
                'Ваш банковский счет приостановлен. Нажмите для возобновления',
                'Вы выиграли 1000 рублей. Запросите сейчас',
                'Ваша кредитная карта заблокирована. Проверьте немедленно',
                'Ваш аккаунт требует срочной проверки',
            ]
        }
        
        # Spam messages in different languages
        spam_messages = {
            'hi': [
                '50% छूट पर सभी उत्पाद। अभी खरीदें',
                'नया मोबाइल फोन मुफ्त में। अभी क्लेम करें',
                'आपका लॉटरी टिकट जीत गया है',
                'सभी ब्रांड्स पर बड़ी छूट',
            ],
            'es': [
                '50% de descuento en todos los productos. Compre ahora',
                'Teléfono móvil gratis. Reclame ahora',
                'Su boleto de lotería ha ganado',
                'Grandes descuentos en todas las marcas',
            ],
            'fr': [
                '50% de réduction sur tous les produits. Achetez maintenant',
                'Téléphone portable gratuit. Réclamez maintenant',
                'Votre billet de loterie a gagné',
                'Grosses réductions sur toutes les marques',
            ],
            'de': [
                '50% Rabatt auf alle Produkte. Jetzt kaufen',
                'Kostenloses Handy. Jetzt einfordern',
                'Ihr Lottoschein hat gewonnen',
                'Große Rabatte auf alle Marken',
            ],
            'zh': [
                '所有产品50%折扣。立即购买',
                '免费手机。立即领取',
                '您的彩票中奖了',
                '所有品牌大减价',
            ],
            'ar': [
                'خصم 50% على جميع المنتجات. اشتر الآن',
                'هاتف محمول مجاني. اطلب الآن',
                'تذكرة اليانصيب الخاصة بك فازت',
                'خصومات كبيرة على جميع العلامات التجارية',
            ],
            'ja': [
                '全商品50%オフ。今すぐ購入',
                '無料携帯電話。今すぐ請求',
                'あなたの宝くじが当選しました',
                '全ブランド大セール',
            ],
            'ko': [
                '모든 제품 50% 할인. 지금 구매하세요',
                '무료 휴대폰. 지금 청구하세요',
                '귀하의 복권이 당첨되었습니다',
                '모든 브랜드 대폭 할인',
            ],
            'ru': [
                '50% скидка на все товары. Покупайте сейчас',
                'Бесплатный мобильный телефон. Запросите сейчас',
                'Ваш лотерейный билет выиграл',
                'Большие скидки на все бренды',
            ]
        }
        
        # Legitimate messages in different languages
        legitimate_messages = {
            'hi': [
                'आपका ऑर्डर डिलीवर हो गया है',
                'आपका बैंक लेनदेन सफल रहा',
                'आपका OTP 123456 है',
                'आपका रिजर्वेशन कन्फर्म हो गया है',
            ],
            'es': [
                'Su pedido ha sido entregado',
                'Su transacción bancaria fue exitosa',
                'Su OTP es 123456',
                'Su reserva ha sido confirmada',
            ],
            'fr': [
                'Votre commande a été livrée',
                'Votre transaction bancaire a réussi',
                'Votre OTP est 123456',
                'Votre réservation a été confirmée',
            ],
            'de': [
                'Ihre Bestellung wurde geliefert',
                'Ihre Banktransaktion war erfolgreich',
                'Ihr OTP ist 123456',
                'Ihre Reservierung wurde bestätigt',
            ],
            'zh': [
                '您的订单已送达',
                '您的银行交易成功',
                '您的验证码是123456',
                '您的预订已确认',
            ],
            'ar': [
                'تم تسليم طلبك',
                'معاملتك المصرفية ناجحة',
                'رمز التحقق الخاص بك هو 123456',
                'تم تأكيد حجزك',
            ],
            'ja': [
                'ご注文が配達されました',
                'お客様の銀行取引が成功しました',
                'あなたのOTPは123456です',
                'ご予約が確認されました',
            ],
            'ko': [
                '주문이 배달되었습니다',
                '귀하의 은행 거래가 성공했습니다',
                '귀하의 OTP는 123456입니다',
                '예약이 확인되었습니다',
            ],
            'ru': [
                'Ваш заказ доставлен',
                'Ваша банковская транзакция прошла успешно',
                'Ваш OTP 123456',
                'Ваше бронирование подтверждено',
            ]
        }
        
        # Create synthetic dataset
        for msg_type, messages in [('fraudulent', fraud_messages.get(language, [])),
                                  ('spam', spam_messages.get(language, [])),
                                  ('legitimate', legitimate_messages.get(language, []))]:
            for message in messages:
                synthetic_data.append({
                    'message': message,
                    'label': msg_type,
                    'language': language
                })
        
        return pd.DataFrame(synthetic_data)
    
    def train_multilingual_models(self, language_data: Dict[str, pd.DataFrame]):
        """
        Train language-specific models
        """
        print("Training multilingual models...")
        
        for lang, df in language_data.items():
            if df.empty:
                print(f"Skipping {self.language_names[lang]} - no data available")
                continue
                
            print(f"Training {self.language_names[lang]} model...")
            
            # Prepare features and labels
            X = df['message'].values
            y = df['label'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words=None,  # Language-specific stop words would be better
                min_df=2,
                max_df=0.95
            )
            
            # Fit and transform training data
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # Train model
            model = MultinomialNB(alpha=1.0)
            model.fit(X_train_tfidf, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{self.language_names[lang]} model accuracy: {accuracy:.3f}")
            print(classification_report(y_test, y_pred))
            
            # Store model and vectorizer
            self.models[lang] = model
            self.vectorizers[lang] = vectorizer
    
    def classify_message(self, text: str) -> Dict[str, any]:
        """
        Classify message using appropriate language model
        """
        # Detect language
        detected_lang = self.detect_language(text)
        
        # Use detected language model if available, otherwise fallback to English
        if detected_lang in self.models:
            lang = detected_lang
        else:
            lang = 'en'
            print(f"Language {detected_lang} not supported, using English model")
        
        # Get model and vectorizer
        model = self.models[lang]
        vectorizer = self.vectorizers[lang]
        
        # Preprocess and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get class labels
        classes = model.classes_
        
        # Create result
        result = {
            'prediction': prediction,
            'language': detected_lang,
            'language_name': self.language_names.get(detected_lang, 'Unknown'),
            'model_used': lang,
            'probabilities': dict(zip(classes, probabilities)),
            'confidence': max(probabilities)
        }
        
        return result
    
    def save_models(self, output_dir: str = "multilingual_models"):
        """
        Save all trained models and vectorizers
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for lang in self.models.keys():
            # Save model
            model_path = os.path.join(output_dir, f"model_{lang}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[lang], f)
            
            # Save vectorizer
            vectorizer_path = os.path.join(output_dir, f"vectorizer_{lang}.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizers[lang], f)
            
            # Save vocabulary
            vocab_path = os.path.join(output_dir, f"vocab_{lang}.json")
            vocabulary = self.vectorizers[lang].vocabulary_
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=2)
        
        # Save language mapping
        lang_map_path = os.path.join(output_dir, "language_mapping.json")
        with open(lang_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.language_names, f, ensure_ascii=False, indent=2)
        
        print(f"Models saved to {output_dir}")
    
    def load_models(self, models_dir: str = "multilingual_models"):
        """
        Load trained models and vectorizers
        """
        if not os.path.exists(models_dir):
            print(f"Models directory {models_dir} not found")
            return
        
        # Load language mapping
        lang_map_path = os.path.join(models_dir, "language_mapping.json")
        if os.path.exists(lang_map_path):
            with open(lang_map_path, 'r', encoding='utf-8') as f:
                self.language_names = json.load(f)
        
        # Load models and vectorizers
        for lang in self.language_names.keys():
            model_path = os.path.join(models_dir, f"model_{lang}.pkl")
            vectorizer_path = os.path.join(models_dir, f"vectorizer_{lang}.pkl")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                with open(model_path, 'rb') as f:
                    self.models[lang] = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers[lang] = pickle.load(f)
                
                print(f"Loaded {self.language_names[lang]} model")
        
        print(f"Loaded {len(self.models)} models")

def main():
    """
    Main function to demonstrate multilingual training
    """
    print("🌍 Multilingual SMS Fraud Detection Training")
    
    # Initialize detector
    detector = MultilingualFraudDetector()
    
    # Test language detection
    test_messages = [
        "Your account has been suspended. Click here to verify",
        "आपका बैंक खाता निलंबित हो गया है",
        "Su cuenta bancaria ha sido suspendida",
        "Votre compte bancaire a été suspendu",
        "Ihr Bankkonto wurde gesperrt",
        "您的银行账户已被暂停",
        "تم تعليق حسابك المصرفي",
        "あなたの銀行口座が停止されました",
        "귀하의 은행 계좌가 일시정지되었습니다",
        "Ваш банковский счет приостановлен",
    ]
    
    print("\n🔍 Testing Language Detection:")
    for message in test_messages:
        detected_lang = detector.detect_language(message)
        print(f"'{message[:30]}...' -> {detector.language_names.get(detected_lang, detected_lang)}")
    
    # Prepare training data (using synthetic data for demonstration)
    print("\n📚 Preparing Training Data:")
    language_data = detector.prepare_multilingual_data("data/sms_dataset.csv")
    
    # Train models
    print("\n🤖 Training Models:")
    detector.train_multilingual_models(language_data)
    
    # Test classification
    print("\n🧪 Testing Multilingual Classification:")
    test_cases = [
        ("Your account has been suspended. Click here to verify", "en"),
        ("आपका बैंक खाता निलंबित हो गया है। क्लिक करें", "hi"),
        ("Su cuenta bancaria ha sido suspendida. Haga clic", "es"),
        ("Votre compte bancaire a été suspendu. Cliquez", "fr"),
        ("Ihr Bankkonto wurde gesperrt. Klicken Sie", "de"),
        ("您的银行账户已被暂停。点击重新激活", "zh"),
        ("تم تعليق حسابك المصرفي. انقر لإعادة التفعيل", "ar"),
        ("あなたの銀行口座が停止されました。クリックしてください", "ja"),
        ("귀하의 은행 계좌가 일시정지되었습니다. 클릭하세요", "ko"),
        ("Ваш банковский счет приостановлен. Нажмите", "ru"),
    ]
    
    for message, expected_lang in test_cases:
        result = detector.classify_message(message)
        print(f"\n{detector.language_names[expected_lang]}:")
        print(f"  Message: {message[:50]}...")
        print(f"  Detected Language: {result['language_name']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.3f}")
    
    # Save models
    print("\n💾 Saving Models:")
    detector.save_models()
    
    print("\n✅ Multilingual training completed!")

if __name__ == "__main__":
    main() 