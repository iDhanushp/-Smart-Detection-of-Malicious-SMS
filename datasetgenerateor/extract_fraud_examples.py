"""
extract_fraud_examples.py
=========================
Standalone script — loads the phone SMS export CSVs, applies the
same label_message() rules used in train_real_model.py, and saves
all rows that were auto-labeled 'fraud' to fraud_training_samples.csv.

Run:
    python extract_fraud_examples.py
"""

import re, os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'sms data set')
OUT_PATH  = os.path.join(os.path.dirname(__file__), 'fraud_training_samples.csv')

# ─── keyword banks (identical to train_real_model.py) ────────────────────────
URGENCY_KW = ['urgent','immediately','asap','expire','deadline','limited time','act now','hurry','last chance','expire today','expires soon','time running out']
FEAR_KW    = ['suspended','blocked','terminated','legal action','penalty','fine','arrest','court','lawsuit','closed','cancelled','frozen','unauthorized']
REWARD_KW  = ['congratulations','winner','won','prize','cash','reward','lottery','jackpot','free','gift','bonus','cashback','refund','lakh','crore']
ACTION_KW  = ['click','call','reply','text','visit','download','verify','confirm','update','provide','share','enter','submit','activate','redeem']

LEGIT_SENDER_BRANDS = {
    'AIRTEL','ARWINF','ARWNFO','ARINFO',
    '650001','650002','650003','650022','650025','650026','602607','602604',
    'JIOVOC','JIOCIN','VODAFO','VODAFN','BSNL','MTNNL','IDEA',
    'SBIINB','SBIPAY','HDFCBK','ICICIB','AXISBK','CANBNK','KOTAKA','KOTAKB',
    'PNBSMS','BOIIND','UNIBAN','SYNDBK','IOBSMS','CBSSMS','YESBNK','IDBIBN',
    'PAYTMB','IPAYTM','PHONPE','GPAYSB','BHIMUPI','RAZPAY','MOBIKW',
    'BAJAJF','BFDLTS','BFDLPS','INDFIN','CAPFST',
    'LICIOF','EXIDLI','SBIINR','ICICIL','HDFCLI',
    'MYGOVT','GOVRAS','KAEOFF','MOBKAR','DIPRUK','RBISAY','TRAI','UIDAI',
    'GOIEMS','NSDLEG','INCOTX','EPFINR','ESICIN',
    'FLPKRT','AMAZON','AMZNIN','MYNTRA','ZOMATO','SWIGGY','887700',
    'IRCTCI','IRCTC','MAKEMY','CLEARR','GOIBOB','IXIGO',
    'DUNZOO','PORTER','BPCLTD','BPCLIN','HPCLIN','IOCLTD',
    'PRACTO','APOLLO','FORTIS','NARAYN',
    'DSCENG','ISBRPG','DSUEDU','LOHITS','JKSHAH','PREUNI','MSRSBG',
    'CRED','NAVI',
    'ANGONE','UTIMFS','NIMFND','NIMFNF','IDBIBK','ONJPTR','JPTR',
    'NSEIND','BSEIND','ZERODH','ZEROD','UPSTOX','GROWW','KUVERA',
    'SBIMFS','HDFCMF','ICICIM','AXISMF','MIRAE','NIPPON','PVRVIP','PVRPAY',
}

SPAM_SENDER_BRANDS = {
    'JERUMY','JLRUMY','JLRMMY','JUNRMY','JRUMMY','JGRUMY','JLERMY',
    'GRRUMY','GARUMY','CLRWEB',
    'PLYWIN','MGLAMM','CRCOIN','SMKKBM',
    'MOPVCW','MPSGWN','FRNDEL','JAINMM','AGIEDU','MOBELC','GOPAUM',
    'ENTERP','ISBRPG','BAPUSE','GETFRE','KOTAKA',
}

TRUSTED_URL_PATTERNS = [
    r'i\.airtel\.in', r'maps\.google\.com', r'splitwise\.com',
    r'whatsapp\.com/dl/', r'bbmpgov\.in', r'utimf\.com',
    r'nipponindia\.com', r'app\.jupiter\.money', r'bflcomm\.in',
    r'pvr\.im', r'amazon\.in/a/c', r'ewsbm\.com',
    r'sindhicollege\.com', r'forms\.gle',
]

# ─── helper functions ────────────────────────────────────────────────────────

def _has_url(t):
    return bool(re.search(r'https?://|www\.|wa\.me/|\.(com|in|org|io|co)', t))

def _has_phone(t):
    return bool(re.search(r'\b\d{10,}\b', t))

def _is_phone_sender(s):
    return bool(re.match(r'^\+\d{10,}$', s))

def _is_service_sender(s):
    return not _is_phone_sender(s) and ('-' in s or len(s) <= 6)

def _has_trusted_url(t):
    return any(re.search(p, t) for p in TRUSTED_URL_PATTERNS)

# ─── label_message (identical to train_real_model.py) ───────────────────────

def label_message(address: str, body: str) -> str:
    s    = address.strip()
    t    = body.lower()
    s_up = s.upper()

    if _is_phone_sender(s):
        if _has_trusted_url(t):
            return 'legit'

        brand_impersonation = [
            r'(?=.*(?:hsbc|barclays))(?=.*(?:rs\s*\d{4,}|member|earned|profits?|returns?|advice|guidance|success|stock\s*(?:market|update)|financial))',
            r'(?=.*(?:indira))(?=.*(?:rs\s*\d{4,}|profits?|stock\s*market|invest|securit|expert\s*advice))',
            r'rbi.{0,20}(?:list(?:ed|ing)|backed?|approved|confirmed)',
            r's\.?b[,\.]?i.{0,5}y[o0]n[o0]',
            r'(?=.*(?:axis\s*bank|hdfc\s*bank|icici\s*bank|kotak\s*bank|sbi\s+(?:bank|personal)))(?=.*(?:https?://|wa\.me/|bit\.ly|epq9|tinyurl))',
            r'(?:bhim|upi).{0,60}(?:withdraw|cash\s*now|get\s*cash)',
            r'(?=.*(?:msme|bima\s*yojana))(?=.*(?:lakh|crore|credited|approved))(?=.*(?:http|\.in/|\.com/))',
            r'(?:goldify|goldifyapp)',
            r'confirm.{0,20}(?:few|your)\s*details.{0,30}http',
            r'transaction.{0,60}rs\.?\s*\d{4,}.{0,60}withdraw',
        ]
        for pat in brand_impersonation:
            if re.search(pat, t, re.IGNORECASE):
                return 'fraud'

        clear_fraud = [
            r'(?:h0me|home).{0,20}(?:j0b|job).{0,60}wa\.me',
            r'(?:video.?kyc|videokyc).{0,60}(?:blocked|activate|account)',
        ]
        for pat in clear_fraud:
            if re.search(pat, t, re.IGNORECASE):
                return 'fraud'

        spam_promo = [
            r'(earn|income|salary).{0,30}(per\s*day|per\s*month|/day|/month|a\s*month)',
            r'(rs\.?\s*\d{3,}|₹\s*\d{3,}).{0,30}(per\s*day|per\s*month|/day|a\s*month)',
            r'(won|winner|selected|lucky|eligible).{0,40}(free\s*gift|purifier|voucher|kent|prize)',
            r'(congratulations|congrats).{0,60}(rummy|casino|poker|fantasy|bonus)',
            r'(register|play|win).{0,30}(rummy|poker|casino|fantasy)',
            r'rummy.{0,40}(bonus|welcome|account|cash)',
            r'paper\s*plate|led\s+bulb|paaku|buyback.{0,20}(raw\s*material|agreement)',
            r'(pi\s+is\s+a\s+new|pi\s+network|minepi\.com)',
            r'(nojoto|talent.{0,20}earn|talent.{0,20}money)',
            r'(car\s+insurance|bike\s+insurance).{0,30}(renew|80%|off)',
            r'n95\s+mask|n95mask',
            r'navratri.{0,60}registr',
        ]
        for pat in spam_promo:
            if re.search(pat, t, re.IGNORECASE):
                return 'spam'

        if re.search(r'(h0me|home).{0,20}(j0b|job)', t):
            return 'spam'
        if _has_url(t):
            return 'spam'
        return 'legit'

    brand = ''
    m = re.match(r'^[A-Z]{2}-(.+)$', s_up)
    if m:
        brand = re.sub(r'-[A-Z]$', '', m.group(1))
    else:
        brand = s_up

    for legit in LEGIT_SENDER_BRANDS:
        if legit in s_up:
            fraud_override = [
                r'(verify|update).{0,30}(aadhaar|kyc|pan).{0,30}(expire|suspend)',
                r'(your account|card).{0,20}\b(suspended|blocked|deactivated)\b.{0,40}(click|link|call)',
            ]
            for pat in fraud_override:
                if re.search(pat, t):
                    return 'fraud'
            return 'legit'

    for sp in SPAM_SENDER_BRANDS:
        if sp in s_up:
            return 'spam'

    if re.search(r'(h0me|home).{0,20}(j0b|job)', t) and re.search(r'wa\.me/', t):
        return 'fraud'

    if re.search(r'(rummy|poker|fantasy|casino|bet|satta)', t) and _has_url(t):
        return 'spam'

    promo_patterns = [
        r'(offer|deal|discount|sale|off).{0,30}(shop|buy|order)',
        r'(download|install).{0,20}(app|apk)',
        r'cashback.{0,20}(order|pay|recharge)',
    ]
    for pat in promo_patterns:
        if re.search(pat, t) and _has_url(t):
            return 'spam'

    fraud_patterns = [
        (r'(?:account|card)\s+\b(blocked|suspended|deactivated)\b'
         r'|\b(?:your\s+)?(?:account|card)\b.{0,40}\b(?:suspended|deactivated)\b', 'account_threat'),
        (r'(kyc|aadhaar|aadhar|pan).{0,40}(expire|expired|update\s*now|verify\s*now|complete\s*kyc|link\s*expire)', 'kyc_fraud'),
        (r'(legal action|court|arrest|fir|police).{0,40}(file|case|action)', 'legal_threat'),
        (r'(unauthorized|suspicious).{0,40}(transaction|login|access)', 'fraud_alert'),
        (r'(income.?tax|irdai|sebi).{0,60}(verify|update|notice)', 'impersonation'),
    ]
    for pat, _ in fraud_patterns:
        if re.search(pat, t):
            return 'fraud'

    if '-' in s or len(s) <= 6:
        return 'legit'
    return 'legit'


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    phone_dfs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith('phone_sms_export') and fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_DIR, fname), dtype=str).fillna('')
            print(f'  Loaded {fname}: {len(df)} rows')
            phone_dfs.append(df)

    if not phone_dfs:
        print('ERROR: no phone_sms_export*.csv files found in', DATA_DIR)
        raise SystemExit(1)

    phone_df = pd.concat(phone_dfs, ignore_index=True)
    phone_df = phone_df.drop_duplicates(subset='body')
    print(f'\n  Total unique messages: {len(phone_df)}')

    phone_df['label'] = phone_df.apply(
        lambda r: label_message(r['address'], r['body']), axis=1
    )

    dist = phone_df['label'].value_counts()
    print(f'  Label distribution:\n{dist}\n')

    fraud_df = phone_df[phone_df['label'] == 'fraud'][['address', 'body', 'label']].reset_index(drop=True)
    fraud_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    print(f'Saved {len(fraud_df)} fraud examples → {OUT_PATH}\n')
    print('─' * 72)
    for i, row in fraud_df.iterrows():
        sender_display = row['address'][:20].ljust(20)
        body_display   = row['body'][:80].replace('\n', ' ')
        print(f'[{i+1:02d}] {sender_display}  {body_display}')
