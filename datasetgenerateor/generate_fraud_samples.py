"""
generate_fraud_samples.py
=========================
Generates ~4000 synthetic Indian fraud SMS samples across 25 categories.
Output: datasetgenerateor/fraud_synthetic_4000.csv  (address, body, label=fraud)

Run:
    python generate_fraud_samples.py
"""

import random, csv, os, string
random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), 'fraud_synthetic_4000.csv')

# ─── helpers ──────────────────────────────────────────────────────────────────
def rphone():
    pfx = random.choice(['70','72','73','74','75','76','77','78','79',
                         '80','81','82','83','84','85','86','87','88','89',
                         '90','91','92','93','94','95','96','97','98','99'])
    return '+91' + pfx + ''.join(random.choices(string.digits, k=8))

def dlt_sender():
    """Fraudulent DLT-style sender codes (not in LEGIT_SENDER_BRANDS)"""
    return random.choice(['CP-MPOKKT','JD-SMKKBM','VM-GETJOB','BG-EARNNW',
                          'TX-LOANOK','VK-KREDTS','BW-INCASH','CP-GOVTSC',
                          'QP-REFUND','TM-CASHNW'])

def ra(lo=1000, hi=500000, step=500):
    """Random Indian Rupee amount"""
    return random.randrange(lo, hi, step)

def racct():
    return 'XXXX' + ''.join(random.choices(string.digits, k=4))

def rdate():
    d = random.choice(['01','03','05','07','10','12','15','17','19','22','25','28'])
    m = random.choice(['01','02','03','04','05','06','07','08','09','10','11','12'])
    y = random.choice(['2025','2026'])
    return f"{d}-{m}-{y}"

def rtime():
    return f"{random.randint(6,22):02d}:{random.choice(['00','15','30','45'])}"

def rref():
    return ''.join(random.choices(string.digits + string.ascii_uppercase, k=10))

def rurl(domains=None):
    if domains is None:
        domains = ['bit.ly','tinyurl.com','rb.gy','t.ly','ow.ly','s.id',
                   'cutt.ly','short.gy','lihi.cc','lnkd.in']
    dom = random.choice(domains)
    path = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5,9)))
    return f'https://{dom}/{path}'

def rphish():
    dom = random.choice([
        'sbi-secure.in','hdfc-kyc.com','icici-verify.net','axis-bank.online',
        'rbi-scheme.in','gov-scheme.in','income-tax-refund.in','paytm-kyc.net',
        'epq9.com','sr3.in','ct3.io','mktasset.com','linstok.com',
        'sbicard-update.com','hdfcloan-apply.in','axisbk-verify.com',
        'icicisecure.net','kotak-kyc.in','yesbank-update.com',
    ])
    path = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(6,14)))
    return f'http://{dom}/{path}'

def rwa():
    return f'wa.me/91{random.randint(7000000000, 9999999999)}'

def rmobile_no():
    pfx = random.choice(['70','72','74','76','78','80','82','84','86','88','90','92','94','96','98'])
    return pfx + ''.join(random.choices(string.digits, k=8))

rows = []
def add(cat, sender, body):
    rows.append((sender, body, 'fraud', cat))

# ══════════════════════════════════════════════════════════════════════════════
# CAT 1 ─ Fake broker / investment advisory  (~300)
# ══════════════════════════════════════════════════════════════════════════════
BROKERS = ['HSBC Securities','BARCLAYS','INDIRA Securities','Morgan Stanley India',
           'Goldman Sachs Advisory','JP Morgan India','Deutsche Securities',
           'Citibank Wealth','UBS India Advisory','Credit Suisse India']
for _ in range(300):
    b = random.choice(BROKERS)
    amt = ra(50000, 600000, 1000)
    url = rurl()
    t = random.choice([
        f"{b} members earned Rs {amt:,} last week. Start building smarter today. {url}",
        f"Achieve Rs {amt:,} in weekly profits with {b}'s trusted advice. {url}",
        f"Reach smarter returns with {b}. Members earned Rs {amt:,} last week. {url}",
        f"India's stock market is rising! Join {b} for expert advice on profitable investments. {url}",
        f"Last week, {b} members earned Rs {amt:,} with expert market insights. {url}",
        f"Reliable strategies from {b} ensured Rs {amt:,} in member profits last week. {url}",
        f"Achieve Rs {amt:,} this week with {b} expert insights. Better returns start today. {url}",
        f"{b} helps you achieve stock market success with real guidance. {url}",
        f"Arbitrage model: from 10K, RBI-backed plan. {b} members earned Rs {amt:,}. {url}",
        f"Join {b} premium group. Members earned Rs {amt:,} last week. Register free: {url}",
        f"Rs {amt:,} profit guaranteed with {b} strategy. Join members now: {url}",
        f"Exclusive: {b} stock tips returned Rs {amt:,} for members this week. {url}",
        f"{b} — proven strategy. Rs {amt:,} weekly returns. Limited seats. {url}",
        f"Don't miss: {b} experts helping members earn Rs {amt:,}/week. Enroll: {url}",
        f"Verified: {b} advisor group. Rs {amt:,} earned by members {rdate()}. {url}",
    ])
    add('fake_broker', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 2 ─ Account blocked / KYC phish  (~220)
# ══════════════════════════════════════════════════════════════════════════════
LOAN_APPS = ['mPokket','KreditBee','MoneyView','CashBean','PaySense','LazyPay',
             'Dhani','Navi Loan','Fibe','SmartCoin']
for _ in range(220):
    app = random.choice(LOAN_APPS)
    url = rphish()
    t = random.choice([
        f"Attention! Account BLOCKED, Complete Video-KYC now to Activate your Account and get a limited-time 50% off on interest rates for your next loan {url} -{app}",
        f"IMPORTANT ALERT! Account BLOCKED due to incomplete re-KYC as per RBI guideline. To activate your account, complete VIDEO-KYC URGENTLY. Loan up to Rs.45,000 instantly {url} -{app}",
        f"Your {app} account is SUSPENDED. Complete KYC immediately to avoid permanent closure. {url}",
        f"URGENT: Your {app} account will be deactivated in 24 hours. Complete Video-KYC now: {url}",
        f"Dear User, your {app} account has been temporarily blocked. Update KYC to continue: {url}",
        f"Action Required! {app}: Your account is blocked due to pending KYC. Click to verify: {url}",
        f"[{app}] ALERT: Account suspended. Complete re-KYC within 2 hours to avoid termination: {url}",
        f"Final Warning from {app}: KYC incomplete. Account will be closed by midnight. Verify now: {url}",
        f"Your {app} loan account is FROZEN. To unfreeze, complete video KYC immediately: {url}",
        f"{app} Notice: Incomplete KYC detected. Your credit limit is on hold. Update here: {url}",
    ])
    sender = random.choice([rphone(), dlt_sender()])
    add('kyc_phish', sender, t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 3 ─ Fake BHIM/UPI credit → withdraw  (~220)
# ══════════════════════════════════════════════════════════════════════════════
UPI_APPS = ['BHIM','Google Pay','PhonePe','Paytm','BHIM SBI','BHIM Axis']
for _ in range(220):
    app = random.choice(UPI_APPS)
    amt = ra(5000, 100000, 500)
    mob = rmobile_no()
    ref = rref()
    url = rphish()
    t = random.choice([
        f"Hi {mob[:4]}XXXX{mob[-2:]}, Rs.{amt:,}/- is Transferred to Your A/c By {app} Ref ID: {ref[:6]}XXX. Withdraw Cash NOW {url}",
        f"Congrats {mob[:4]}XXXX{mob[-2:]}, Rs.{amt:,}/- Credited to your A/c By {app} Ref ID: {ref[:6]}XXX GET Cash NOW {url}",
        f"Hi {mob[:4]}XXXX{mob[-2:]}, Transaction successfully done of Rs.{amt:,} to your account on {rdate()}. Withdraw N0W: {url}",
        f"Rs.{amt:,} credited to your account via {app}. Ref: {ref[:8]}. To withdraw click: {url}",
        f"Received INR {amt:,} on {mob[:6]}XX{mob[-2:]} at {rdate()} {rtime()}. Quick UPI withdrawal available here: {url}",
        f"[{app}] Rs {amt:,} has been sent to your UPI account. Withdraw instantly here: {url}",
        f"Transaction alert: Rs.{amt:,} credited by {app} to {mob[:4]}xxxx{mob[-4:]}. Redeem now: {url}",
        f"Your {app} wallet received Rs.{amt:,}. Claim and withdraw: {url}",
        f"BHIM Ref ID: {ref}. Rs.{amt:,} sent to {mob[:4]}XXXX. Withdraw Cash NOW {url}",
        f"UPI Transfer: INR {amt:,} received on {mob[:4]}XXXX{mob[-4:]} from government scheme. Withdraw: {url}",
    ])
    add('bhim_upi_fake', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 4 ─ Fake bank loan solicitation from phone number  (~180)
# ══════════════════════════════════════════════════════════════════════════════
BANKS = ['SBI','HDFC BANK','AXIS BANK','ICICI BANK','KOTAK BANK','YES BANK',
         'PNB','CANARA BANK','UNION BANK','BANK OF BARODA']
LOAN_TYPES = ['Personal Loan','Home Loan','Business Loan','Car Loan','Education Loan','Gold Loan']
for _ in range(180):
    bank = random.choice(BANKS)
    loan = random.choice(LOAN_TYPES)
    rate = round(random.uniform(9.5, 14.5), 2)
    mob = rmobile_no()
    url = random.choice([rphish(), rwa()])
    t = random.choice([
        f"{bank} {loan} Interest Rates Starts at {rate}%. Faster Process & Door Step Service. Apply! {mob} T&C {url}",
        f"{bank} {loan} now rate starts at {rate}%. Faster process, Zero Preclosure charges. Apply! {mob} Whatsapp {url}",
        f"Check your eligibility in 45 sec for {bank} {loan}. Get welcome benefits up to Rs.500. T&C {url}",
        f"Pre-approved {loan} offer for you from {bank}. Rs.5 Lakh in 24 hrs. Apply: {url}",
        f"Congratulations! You are pre-selected for a {bank} {loan} at {rate}% p.a. Limited time offer. {url}",
        f"URGENT: {bank} {loan} offer expires today. Low rate {rate}%. No documents needed. {url}",
        f"{bank} is offering {loan} upto Rs.50 Lakh at just {rate}% p.a. Fast approval. Apply via {url}",
        f"Get instant {bank} {loan} approval in minutes. Rate {rate}%. Call {mob} or apply: {url}",
        f"Special offer: {bank} {loan} with 0 processing fee. Apply now before offer ends: {url}",
        f"You qualify for {bank} {loan} Rs 2 Lakh to Rs 25 Lakh. No collateral. {url}",
    ])
    add('fake_loan', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 5 ─ Fake delivery / courier scam  (~200)
# ══════════════════════════════════════════════════════════════════════════════
COURIERS = ['India Post','DTDC','Blue Dart','Delhivery','Ekart','XpressBees',
            'Ecom Express','Amazon Logistics','Flipkart Courier','FedEx India']
for _ in range(200):
    courier = random.choice(COURIERS)
    track = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    url = rurl(['lihi.cc','bit.ly','tinyurl.com','rb.gy','s.id'])
    amt = ra(50, 300, 10)
    t = random.choice([
        f"Kindly update your delivery location within 12 hours, otherwise we will proceed to return the product: {url}",
        f"{courier}: Due to an incorrect house number, we are unable to deliver your package. Please update your address: {url}",
        f"Post Office: Your parcel {track} is on hold due to incomplete address. Update now: {url}",
        f"Your {courier} package could not be delivered. Reschedule delivery here: {url}",
        f"NOTICE: Your parcel is waiting at the warehouse. Pay Rs.{amt} shipping fee to release: {url}",
        f"{courier} Alert: Package {track} requires address confirmation. Update within 24 hrs: {url}",
        f"Your shipment has been held at customs. Pay Rs.{amt} customs duty to release: {url}",
        f"Dear Customer, your {courier} parcel failed delivery attempt. Reschedule: {url}",
        f"Final attempt: {courier} was unable to deliver your package. Click to reschedule: {url}",
        f"[{courier}] Tracking {track}: Package undeliverable. Update delivery address: {url}",
        f"Your order from Amazon is on hold. Confirm delivery address within 6 hours: {url}",
        f"Flipkart Delivery: We attempted delivery but found the address incorrect. Update: {url}",
        f"Customs duty of Rs.{amt} is pending for your international parcel. Pay now: {url}",
        f"India Post: Parcel {track} detained. Pay Rs.{amt} handling fee to release: {url}",
        f"Your package is at the nearest post office. Pay Rs.{amt} to receive it: {url}",
    ])
    add('fake_delivery', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 6 ─ Fake bank credit alert / phishing from phone number  (~200)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(200):
    bank = random.choice(BANKS)
    amt = ra(1000, 50000, 500)
    acct = racct()
    url = rphish()
    t = random.choice([
        f"Dear Customer, Rs.{amt:,}/- has been Credited to your {bank} A/c {acct} on {rdate()}. If not done by you, call or click {url}",
        f"{bank} ALERT: Rs.{amt:,} credited to A/c ending {acct[-4:]}. If unrecognized, verify immediately: {url}",
        f"Congratulations! Rs.{amt:,}/- credited to your {bank} account by GOVT scheme. Claim now: {url}",
        f"[{bank}] Rs.{amt:,} has been CREDITED to {acct}. Confirm the transaction: {url}",
        f"{bank}: An unusual transaction of Rs.{amt:,} detected. To block/confirm: {url}",
        f"Alert: Rs.{amt:,} debited from {bank} A/c {acct} at {rtime()} on {rdate()}. Not you? {url}",
        f"Your {bank} account {acct} received Rs.{amt:,}. To withdraw or verify: {url}",
        f"{bank} — Your KYC-linked reward of Rs.{amt:,} is ready. Claim before expiry: {url}",
        f"NOTICE: Rs.{amt:,} reversal pending in {bank} account {acct}. Confirm: {url}",
        f"Your {bank} cashback of Rs.{amt:,} is ready to be transferred. Verify account: {url}",
    ])
    add('fake_bank_credit', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 7 ─ RBI / govt investment scheme  (~150)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(150):
    amt = ra(10000, 5000000, 5000)
    url = random.choice([rphish(), rurl()])
    t = random.choice([
        f"Rs{ra(10000,100000,1000)} to Rs{amt:,} story. RBI listing confirmed. {url}",
        f"RBI-backed arbitrage plan: invest Rs.10K, earn Rs.{amt:,}/month. Limited slots. {url}",
        f"Govt approved investment scheme: Rs.{amt:,} guaranteed return. Apply now: {url}",
        f"SEBI-registered fund offers {random.randint(15,45)}% annual return. Invest Rs.10K today. {url}",
        f"PM Digital India Fund: Rs.{amt:,} credited for eligible citizens. Claim: {url}",
        f"RBI Digital Saving Scheme 2026: Invest Rs.5K get Rs.{amt:,} in 90 days. {url}",
        f"Ministry of Finance approved: earn Rs.{amt:,}/week. Register today: {url}",
        f"MSME Loan Scheme 2026 - Rs.{amt:,}/- can be credited in your Bank. Check eligibility: {url}",
        f"Dear citizen, Rs.{amt:,} from PM Kisan fund is ready for withdrawal. Verify: {url}",
        f"Bima Yojana: You are selected to get 1 CRORE life cover. Application approved. Apply: {url}",
        f"Your name appears in RBI unclaimed deposit list: Rs.{amt:,}. Claim now: {url}",
        f"GOI Digital Dividend: Rs.{amt:,} allocated to your Aadhaar. Redeem: {url}",
        f"Pradhan Mantri Yojana: Rs.{amt:,} grant approved for your family. Click: {url}",
    ])
    add('rbi_govt_scheme', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 8 ─ OTP / credential theft  (~150)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(150):
    bank = random.choice(BANKS)
    otp = ''.join(random.choices(string.digits, k=6))
    url = rphish()
    t = random.choice([
        f"Your {bank} OTP is {otp}. DO NOT share with anyone. If you didn't request this, report: {url}",
        f"[{bank}] OTP for fund transfer: {otp}. Valid 10 mins. Share this OTP with our executive to complete KYC.",
        f"Dear Customer, your OTP {otp} for updating net banking is valid for 5 min. Call 1800XXXXXX if not requested.",
        f"Security alert: Someone is trying to access your {bank} account. Verify with OTP {otp} here: {url}",
        f"[BANK SECURITY] Your account is being accessed. Enter OTP {otp} to block transaction: {url}",
        f"Your ATM PIN has been reset. New OTP: {otp}. Call our helpdesk to confirm: {url}",
        f"Action required: Enter OTP {otp} to prevent unauthorized withdrawal from {bank} account: {url}",
        f"Your {bank} net banking is locked. Use OTP {otp} to unlock via: {url}",
        f"Final OTP to complete {bank} KYC verification: {otp}. Share with bank executive on call.",
        f"[UPI] Transaction of Rs.{ra(1000,50000,500):,} pending. Confirm with OTP {otp}: {url}",
    ])
    add('otp_theft', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 9 ─ Electricity / gas bill disconnection threat  (~150)
# ══════════════════════════════════════════════════════════════════════════════
UTILITIES = ['BESCOM','MSEDCL','TNEB','CESC','BSES','TATA Power','Adani Electricity',
             'Indraprastha Gas','Mahanagar Gas','GAIL Gas']
for _ in range(150):
    util = random.choice(UTILITIES)
    amt = ra(500, 8000, 100)
    mob = rmobile_no()
    t = random.choice([
        f"Dear Consumer, your {util} electricity connection will be disconnected tonight at 9:30 PM due to non-payment of Rs.{amt:,}. Pay now or call {mob}.",
        f"URGENT: Your electricity supply will be disconnected in 2 hours. Outstanding bill Rs.{amt:,}. Pay: {rphish()}",
        f"{util} Notice: Rs.{amt:,} electricity bill overdue. To avoid disconnection call our executive: {mob}",
        f"Your {util} meter will be blocked today. Pay Rs.{amt:,} immediately to avoid disruption. Helpline: {mob}",
        f"Alert: {util} disconnection scheduled for tonight. Bill due Rs.{amt:,}. Avoid by calling {mob} now.",
        f"FINAL WARNING: {util} will disconnect power supply at {rtime()} today. Pending bill Rs.{amt:,}. {rphish()}",
        f"Your gas supply will be SUSPENDED tonight due to Rs.{amt:,} outstanding. Contact: {mob}",
        f"[{util}] Your account has been flagged. Pay Rs.{amt:,} or face legal action. Call: {mob}",
        f"Mahanagar Gas: Connection termination scheduled. Clear dues Rs.{amt:,} by calling: {mob}",
        f"Electric meter tampering detected. Fine Rs.{amt:,}. Avoid FIR by calling helpline: {mob}",
    ])
    add('utility_threat', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 10 ─ Fake prize / lottery winner  (~180)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(180):
    prize = ra(50000, 5000000, 10000)
    mob = rmobile_no()
    url = random.choice([rphish(), rwa()])
    t = random.choice([
        f"Congratulations! You have WON Rs.{prize:,}/- in the Lucky Draw. To claim your prize, contact: {mob}",
        f"You are selected as the LUCKY WINNER of Rs.{prize:,} from KBC. Call now: {mob}",
        f"WINNER ALERT: Your mobile number has won Rs.{prize:,} in Jio Lucky Draw. Claim: {url}",
        f"Airtel Lucky Winner: Your number won Rs.{prize:,}. Claim before 48 hrs: {mob}",
        f"Dear User, your SIM has been selected in BSNL Lottery. Prize: Rs.{prize:,}. Claim: {mob}",
        f"AMAZON LUCKY DRAW: You won a prize of Rs.{prize:,}. To claim WhatsApp: {rwa()}",
        f"You have won an iPhone 15 + Rs.{ra(10000,50000,1000):,} cash. Confirm your address: {url}",
        f"Flipkart Big Billion Lucky Winner: Rs.{prize:,} + gifts. Collect: {url}",
        f"Your mobile number was picked in GOVT Digital India Lottery. Prize: Rs.{prize:,}. {mob}",
        f"RBI Digital Reward: Rs.{prize:,} allocated to your number. Claim via: {url}",
        f"Congratulations! SPIN and WIN: You won Rs.{prize:,}. Claim in 24 hrs: {url}",
        f"You are a WINNER! Rs.{prize:,} transferred pending your confirmation. Call {mob} now.",
    ])
    add('lottery_prize', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 11 ─ Part-time job / WFH scam (WhatsApp redirect)  (~200)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(200):
    per_day = ra(2000, 10000, 500)
    per_month = per_day * 30
    wa = rwa()
    t = random.choice([
        f"Work from home opportunity! Earn Rs.{per_day:,}/day. No experience needed. WhatsApp: {wa}",
        f"Part time job available. Earn Rs.{per_month:,}/month doing simple tasks. Join: {wa}",
        f"Online data entry job. Earn Rs.{per_day:,} daily. 2-3 hours/day. Contact: {wa}",
        f"Home-based job opportunity. Earn Rs.{per_day:,}/day by liking YouTube videos. {wa}",
        f"Simple online tasks. Rs.{per_day:,}/day. Work from home. WhatsApp now: {wa}",
        f"Hiring now: Part-time online workers. Rs.{per_month:,}/month guaranteed. {wa}",
        f"H0me based J0B available. Earn Rs.{per_day:,} per day. wa.me join: {wa}",
        f"URGENT HIRING: work from home, earn Rs.{per_day:,}/day. Limited seats. {wa}",
        f"Online earning opportunity: Rs.{per_day:,}/day. No investment. Join us: {wa}",
        f"Earn Rs.{per_month:,} monthly from home. Flexible hours. Register: {wa}",
        f"Video rating job: earn Rs.{per_day:,}/day. 1-2 hrs work. WhatsApp: {wa}",
        f"Amazon product review job from home. Earn Rs.{per_day:,}/task. {wa}",
    ])
    add('wfh_job_scam', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 12 ─ Income tax refund fraud  (~120)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(120):
    refund = ra(2000, 50000, 500)
    url = rphish()
    t = random.choice([
        f"Income Tax Department: Your IT refund of Rs.{refund:,} is approved. Update bank account: {url}",
        f"CBDT Notice: Rs.{refund:,} tax refund pending for PAN XXXXX1234X. Claim now: {url}",
        f"Your Income Tax return for FY 2024-25 has a refund of Rs.{refund:,}. Verify: {url}",
        f"IT Dept: Refund of Rs.{refund:,} initiated to your account. Confirm bank details: {url}",
        f"URGENT: Your income tax refund Rs.{refund:,} will lapse in 48 hrs. Update: {url}",
        f"Tax refund alert: Rs.{refund:,} ready. Submit bank details to receive: {url}",
        f"Income Tax refund Rs.{refund:,} has been processed. Update Aadhaar-linked A/c: {url}",
        f"Dear taxpayer, your e-filing refund of Rs.{refund:,} is pending. Claim: {url}",
        f"Income Tax India: Excess TDS of Rs.{refund:,} refundable. Click to claim: {url}",
        f"NOTICE: Your IT refund Rs.{refund:,} cannot be processed — KYC mismatch. Update: {url}",
    ])
    add('income_tax_fraud', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 13 ─ Aadhaar / PAN linking fraud  (~150)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(150):
    bank = random.choice(BANKS)
    url = rphish()
    mob = rmobile_no()
    t = random.choice([
        f"UIDAI Notice: Your Aadhaar is not linked to {bank}. Link before {rdate()} to avoid account freeze: {url}",
        f"Dear Customer, link Aadhaar to your {bank} account immediately. Deadline: {rdate()}. {url}",
        f"Alert: Your PAN is not linked to Aadhaar. Account will be suspended. Update: {url}",
        f"NSDL: Your PAN card will be deactivated on {rdate()}. Link with Aadhaar now: {url}",
        f"RBI Mandate: Aadhaar-Bank linking mandatory by {rdate()}. Update to avoid freeze: {url}",
        f"Aadhaar OTP required to complete {bank} e-KYC. Call {mob} to proceed.",
        f"Income Tax Dept: PAN-Aadhaar linking pending. Penalty Rs.1000/day after {rdate()}. {url}",
        f"Your {bank} account will be FROZEN if Aadhaar not linked by tonight. Link: {url}",
        f"UIDAI: Biometric mismatch on your Aadhaar. Update within 24 hrs: {url}",
        f"Dear Sir/Madam, Aadhaar verification failed for your {bank} account. Re-verify: {url}",
    ])
    add('aadhaar_pan_fraud', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 14 ─ SIM card expiry / upgrade fraud  (~100)
# ══════════════════════════════════════════════════════════════════════════════
TELECOMS = ['Jio','Airtel','Vi (Vodafone Idea)','BSNL']
for _ in range(100):
    tc = random.choice(TELECOMS)
    mob = rmobile_no()
    url = rphish()
    t = random.choice([
        f"Dear {tc} user, your SIM will expire today. Call {mob} immediately to upgrade to 5G SIM.",
        f"URGENT: Your {tc} SIM card is outdated. Upgrade to 5G now or lose number: {url}",
        f"{tc} Alert: Your current SIM will be deactivated tonight. Call {mob} to retain number.",
        f"Your {tc} number will be cancelled due to KYC mismatch. Re-verify now: {url}",
        f"[{tc}] SIM upgrade required. Failure to upgrade by midnight will deactivate your number. {mob}",
        f"Dear {tc} subscriber, new TRAI regulation requires SIM re-registration. Call: {mob}",
        f"{tc} Final Notice: Your mobile number will be blocked by {rtime()}. Upgrade SIM: {mob}",
        f"TRAI order: {tc} users must submit fresh KYC by {rdate()}. Update: {url}",
        f"{tc}: We noticed SIM cloning attempt. Secure your number immediately. Call: {mob}",
        f"Your {tc} SIM is flagged for suspicious activity. To avoid block call: {mob}",
    ])
    add('sim_expiry_fraud', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 15 ─ Police / CBI / court impersonation  (~100)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(100):
    mob = rmobile_no()
    fine = ra(5000, 50000, 1000)
    t = random.choice([
        f"CBI Notice: A case has been registered against your Aadhaar number. Call {mob} immediately to avoid arrest.",
        f"Cyber Crime Police: Your IP address has been traced to illegal activity. Pay Rs.{fine:,} settlement. Call {mob}.",
        f"TRAI Notice: Your number is being used for criminal activity. It will be blocked in 2 hours. Call {mob}.",
        f"Delhi Cyber Crime: FIR filed against your mobile number. Report to station or call {mob} to settle.",
        f"Ministry of Home Affairs: Your digital activity is under investigation. Cooperate: {mob}",
        f"NOTICE from Court of India: Non-bailable warrant issued in your name. Call {mob} for settlement.",
        f"ED Notice: Money laundering case linked to your Aadhaar. Pay Rs.{fine:,} or face arrest. Call {mob}.",
        f"Narcotics Department: Your parcel contains banned items. Pay Rs.{fine:,} fine. Call: {mob}",
        f"Supreme Court Summons: Your PAN is linked to a fraud case. Respond within 2 hrs: {mob}",
        f"CBI Digital Crime Unit: You are under digital surveillance. Cooperate by calling {mob} now.",
    ])
    add('police_impersonation', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 16 ─ Credit card block / unauthorized transaction  (~150)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(150):
    bank = random.choice(BANKS)
    amt = ra(1000, 80000, 500)
    url = rphish()
    t = random.choice([
        f"{bank} Credit Card Alert: Rs.{amt:,} transaction on your card ending {racct()[-4:]}. Not you? Block: {url}",
        f"ALERT: {bank} card used for Rs.{amt:,} online purchase at {rtime()} today. Verify: {url}",
        f"Suspicious transaction of Rs.{amt:,} on {bank} card. To block click: {url}",
        f"{bank}: Your credit card {racct()} has been temporarily blocked. To unblock: {url}",
        f"Unauthorized transaction detected on your {bank} card: Rs.{amt:,}. Report: {url}",
        f"{bank} Card Services: EMI due of Rs.{amt:,} overdue. Pay to avoid legal action: {url}",
        f"URGENT: {bank} credit card about to be cancelled. Verify identity to retain card: {url}",
        f"{bank}: Your card has been cloned. Secure it now by clicking: {url}",
        f"Transaction declined for Rs.{amt:,} on {bank} card. Confirm details: {url}",
        f"{bank} OTP for credit card transaction Rs.{amt:,}. If not you, click to cancel: {url}",
    ])
    add('credit_card_fraud', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 17 ─ Fake Amazon / Flipkart order scam  (~150)
# ══════════════════════════════════════════════════════════════════════════════
PLATFORMS = ['Amazon','Flipkart','Meesho','Myntra','Snapdeal','Nykaa']
PRODUCTS = ['iPhone 15','Samsung S24','laptop','TV','refrigerator','smart watch',
            'earbuds','tablet','air conditioner','washing machine']
for _ in range(150):
    platform = random.choice(PLATFORMS)
    prod = random.choice(PRODUCTS)
    track = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    amt = ra(200, 5000, 100)
    url = rphish()
    t = random.choice([
        f"{platform}: Your order for {prod} (Order #{track}) is on hold. Pay Rs.{amt} to release: {url}",
        f"Dear Customer, {platform} order cancelled due to payment failure. Re-confirm: {url}",
        f"{platform} Alert: Suspicious activity on your account. Verify to continue shopping: {url}",
        f"Your {platform} account has been locked. Unlock by verifying identity: {url}",
        f"IMPORTANT: {platform} refund of Rs.{ra(500,5000,100):,} pending. Confirm bank account: {url}",
        f"{platform} Prize: You've been selected for a free {prod}! Claim before 24 hrs: {url}",
        f"COD order on {platform} worth Rs.{ra(1000,20000,500):,} placed from your account. Not you? {url}",
        f"{platform}: Your {prod} delivery requires payment of Rs.{amt} customs. Pay: {url}",
        f"Flash reward from {platform}: Rs.{ra(200,2000,100):,} cashback waiting. Claim: {url}",
        f"Urgent: {platform} seller issued refund Rs.{ra(500,5000,100):,}. To receive, update account: {url}",
    ])
    add('fake_ecom', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 18 ─ Crypto / trading app investment scam  (~150)
# ══════════════════════════════════════════════════════════════════════════════
CRYPTO_APPS = ['Goldify','CryptoEdge','BitProfit India','CoinMax','TradeMaster',
               'ForexPro India','StockMaster','AlgoTrade24','ProfitEdge']
for _ in range(150):
    app = random.choice(CRYPTO_APPS)
    amt = ra(5000, 500000, 1000)
    url = random.choice([rphish(), rurl()])
    t = random.choice([
        f"Register: {url} ? {app} Golden Opportunity to make Rs.{amt:,}+ income in a Day. Join now!",
        f"!$! {app} DEALER ACCOUNT — Make Rs.{amt:,}/month guaranteed. Register: {url}",
        f"{app}: Your crypto portfolio grew {random.randint(20,300)}% this week. Continue: {url}",
        f"Invest Rs.10,000 in {app} and earn Rs.{amt:,} in 30 days. Verified traders. {url}",
        f"{app} — Top trading signals. Members earned Rs.{amt:,} this month. Join: {url}",
        f"BITCOIN opportunity: Double your investment in 48 hrs with {app}. Limited: {url}",
        f"Passive income from {app}: Rs.{amt:,}/week. No trading experience needed. {url}",
        f"USDT trading signal group — earn Rs.{amt:,}/day. Free join: {url}",
        f"Crypto arbitrage: Rs.10K → Rs.{amt:,} in 7 days. {app} verified. Join: {url}",
        f"{app} is offering 200% returns on crypto trades. Invest now: {url}",
    ])
    add('crypto_scam', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 19 ─ Insurance fraud / fake renewal  (~100)
# ══════════════════════════════════════════════════════════════════════════════
INSURERS = ['LIC','HDFC Life','SBI Life','ICICI Prudential','Max Life',
            'Bajaj Allianz','Tata AIA','Kotak Life']
for _ in range(100):
    ins = random.choice(INSURERS)
    premium = ra(2000, 20000, 500)
    bonus = ra(5000, 100000, 1000)
    mob = rmobile_no()
    url = rphish()
    t = random.choice([
        f"{ins}: Your policy will lapse in 24 hrs. Pay Rs.{premium:,} now to avoid coverage gap: {url}",
        f"ALERT: {ins} policy overdue. Last chance to pay Rs.{premium:,} premium. Call {mob}.",
        f"Dear Policyholder, your {ins} bonus of Rs.{bonus:,} is ready. Claim before expiry: {url}",
        f"{ins}: Your policy matured. Rs.{bonus:,} waiting. Provide bank details: {url}",
        f"IRDAI Notice: {ins} policy must be renewed by {rdate()}. Renew: {url}",
        f"Your {ins} policy claim of Rs.{bonus:,} approved. Collect by {rdate()}: {url}",
        f"Dear customer, Rs.{premium:,} {ins} premium bounced. Pay now to avoid lapse: {mob}",
        f"{ins} Agent: Special 80% discount on life cover. Offer ends today. Call: {mob}",
        f"Rs.{bonus:,} survival benefit from {ins} ready. Update NEFT details: {url}",
        f"IRDAI approved: {ins} double cover scheme. Enroll now before deadline: {url}",
    ])
    add('insurance_fraud', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 20 ─ EMI overdue / loan recovery threat  (~100)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(100):
    bank = random.choice(BANKS)
    emi = ra(2000, 20000, 500)
    mob = rmobile_no()
    url = rphish()
    t = random.choice([
        f"LEGAL NOTICE: Your {bank} loan EMI of Rs.{emi:,} is overdue. Pay immediately or face legal action. Call {mob}.",
        f"{bank} Recovery: Your EMI of Rs.{emi:,} is {random.randint(1,6)} months overdue. Settle or court notice will be issued. {mob}",
        f"Dear borrower, {bank} loan account flagged for non-payment. Pay Rs.{emi:,} now: {url}",
        f"Final notice: {bank} will file case for unpaid loan. Settle Rs.{emi:,} today: {mob}",
        f"Bajaj Finance: Your EMI is pending. Pay Rs.{emi:,} to avoid credit score damage: {url}",
        f"[{bank}] Loan account {racct()} has missed {random.randint(2,5)} EMIs. Regularize immediately: {mob}",
        f"CIBIL Alert: Your credit score dropping due to {bank} overdue. Pay Rs.{emi:,}: {url}",
        f"{bank} Legal Team: Non-payment of Rs.{emi:,} EMI may result in property attachment. Call {mob}.",
        f"Recovery Agent Notice: {bank} has authorized recovery of Rs.{emi*random.randint(2,6):,}. Contact: {mob}",
        f"URGENT: {bank} NACH debit of Rs.{emi:,} failed. Pay manually to avoid penalty: {url}",
    ])
    add('emi_threat', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 21 ─ Fake PM / Govt scheme  (~150)
# ══════════════════════════════════════════════════════════════════════════════
SCHEMES = ['PM Kisan Yojana','PM Ujjwala Yojana','PM Awas Yojana','Ayushman Bharat',
           'Sukanya Samriddhi','Atal Pension Yojana','PM Garib Kalyan','Digital India Fund']
for _ in range(150):
    scheme = random.choice(SCHEMES)
    amt = ra(5000, 200000, 5000)
    url = rphish()
    mob = rmobile_no()
    t = random.choice([
        f"Dear beneficiary, your {scheme} payment of Rs.{amt:,} is approved. Claim: {url}",
        f"Govt of India: Rs.{amt:,} sanctioned under {scheme}. Last date {rdate()}. Apply: {url}",
        f"NOTICE: Your {scheme} funds of Rs.{amt:,} are uncollected. Claim before {rdate()}: {url}",
        f"{scheme}: You are eligible for Rs.{amt:,} grant. Register now: {url}",
        f"PM Office: {scheme} cash benefit Rs.{amt:,} credited to your account. Verify: {url}",
        f"Central Govt Grant: Rs.{amt:,} allocated under {scheme}. Apply today: {url}",
        f"[GOVT] Your application for {scheme} approved. Get Rs.{amt:,}. Confirm Aadhaar: {url}",
        f"Dear Citizen, Rs.{amt:,} NEFT pending under {scheme}. Provide bank details: {mob}",
        f"{scheme} 2026: New beneficiaries selected. Your name found. Collect Rs.{amt:,}: {url}",
        f"Ministry of Finance: {scheme} beneficiary — Rs.{amt:,} on hold. Submit KYC: {url}",
    ])
    add('fake_govt_scheme', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 22 ─ SBI YONO / net banking phishing  (~100)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(100):
    pts = ra(1000, 15000, 100)
    url = rphish()
    t = random.choice([
        f"Dear Customer-Your S.B,I Y0N0 NetBanking Reward point's INR.{pts:,} will expire Today please-redeem Your points in Cash-click-on link. {url}",
        f"SBI YONO: Your reward points {pts:,} expire in 24 hours. Redeem now: {url}",
        f"Dear SBI customer, Rs.{pts:,} cashback from YONO expires today. Click to credit: {url}",
        f"[SBI YONO] Your account has been suspended. Login to verify: {url}",
        f"SBI Net Banking: Your login credentials expire soon. Update password: {url}",
        f"ALERT: Suspicious login attempt on SBI YONO. Secure account: {url}",
        f"SBI: Your internet banking will be blocked in 24 hrs. Confirm identity: {url}",
        f"YONO SBI: Unusual transaction detected. Verify your account to continue: {url}",
        f"SBI Net Banking locked. Provide OTP to unlock via: {url}",
        f"SBI YONO: Your debit card will be blocked. Prevent it by verifying: {url}",
    ])
    add('sbi_yono_phish', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 23 ─ Fake job offer (corporate impersonation)  (~150)
# ══════════════════════════════════════════════════════════════════════════════
COMPANIES = ['TCS','Infosys','Wipro','Amazon India','Google India','Microsoft India',
             'Flipkart','Accenture','HCL','Cognizant']
for _ in range(150):
    co = random.choice(COMPANIES)
    sal = ra(20000, 80000, 2000)
    mob = rmobile_no()
    url = random.choice([rphish(), rwa()])
    t = random.choice([
        f"Congratulations! {co} has shortlisted you for a job. Salary: Rs.{sal:,}/month. Confirm: {mob}",
        f"Job offer from {co}: Rs.{sal:,}/month. Work from home. Registration fee Rs.{ra(500,2000,100)}. {url}",
        f"You have been selected for {co} online job vacancy. Salary Rs.{sal:,}. Joining fee applies. {mob}",
        f"Dear Candidate, {co} HR team found your profile. Interview scheduled. Confirm: {mob}",
        f"HIRING ALERT: {co} needs 50 online workers. Rs.{sal:,}/month. Apply: {url}",
        f"{co} Recruitment 2026: Walk-in-interview. Earn Rs.{sal:,}/month. Register: {url}",
        f"Your resume has been selected by {co}. Rs.{sal:,} salary. Call HR: {mob}",
        f"Online job at {co}: data entry, social media tasks. Rs.{sal:,}/month. Join: {url}",
        f"{co} is hiring freshers. Rs.{sal:,} CTC. Immediate joining. Registration: {url}",
        f"Part time with {co}: Rs.{ra(500,3000,100)}/hour. Work 3 hrs/day. Apply: {url}",
    ])
    add('fake_job_offer', rphone(), t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 24 ─ Aadhaar / KYC update from DLT sender (non-legit)  (~100)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(100):
    bank = random.choice(BANKS)
    url = rphish()
    sender = dlt_sender()
    t = random.choice([
        f"Dear Customer, your {bank} KYC will expire on {rdate()}. Update Aadhaar to avoid account suspension: {url}",
        f"ALERT: {bank} account KYC incomplete. Complete Video-KYC urgently: {url}",
        f"{bank} KYC expired. Account restricted. Update at: {url}",
        f"Final notice: {bank} Aadhaar-linked KYC pending. Penalty applies after {rdate()}: {url}",
        f"RBI directive: Complete {bank} re-KYC by {rdate()} to avoid account freeze: {url}",
        f"Your {bank} account is partially frozen. Resume service by completing KYC: {url}",
        f"{bank} Notice: PAN verification failed. KYC update required: {url}",
        f"Account operations restricted on {bank}. Submit KYC documents: {url}",
        f"URGENT: {bank} account will be closed in 48 hrs due to incomplete KYC: {url}",
        f"[{bank}] Compliance: Update e-KYC before {rdate()} to avoid deactivation: {url}",
    ])
    add('dlt_kyc_phish', sender, t)

# ══════════════════════════════════════════════════════════════════════════════
# CAT 25 ─ Fake property / real estate fraud  (~80)
# ══════════════════════════════════════════════════════════════════════════════
for _ in range(80):
    price = ra(500000, 5000000, 50000)
    mob = rmobile_no()
    url = random.choice([rphish(), rwa()])
    t = random.choice([
        f"Buy flat in Bangalore: Rs.{price:,}. Registry done in 2 days. No broker. Call {mob}",
        f"Plot available near Hyderabad highway. Rs.{price:,}. Direct owner. WhatsApp: {rwa()}",
        f"URGENT SALE: 2BHK flat for Rs.{price:,}. Owner going abroad. Immediate registry. {mob}",
        f"Affordable flats near IT park. Rs.{price:,}. Book with Rs.{ra(5000,20000,1000):,} token. {url}",
        f"GOVT housing scheme: 1BHK for Rs.{ra(200000,600000,10000):,}. Apply online: {url}",
        f"PM Awas Yojana: Subsidized flat in your city. Apply before {rdate()}: {url}",
        f"Land for sale: {random.randint(1,10)} acres near {random.choice(['Pune','Chennai','Indore','Jaipur'])}. Rs.{price:,}. Call {mob}",
        f"Cheap property grab: Bank auction flat Rs.{price:,}. First come first serve. {mob}",
    ])
    add('real_estate_fraud', rphone(), t)

# ─── write CSV ────────────────────────────────────────────────────────────────
random.shuffle(rows)
with open(OUT_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['address', 'body', 'label', 'category'])
    writer.writerows(rows)

# ─── summary ──────────────────────────────────────────────────────────────────
from collections import Counter
cat_counts = Counter(r[3] for r in rows)
print(f"\nSaved {len(rows)} rows → {OUT_PATH}\n")
print("Category breakdown:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<30} {cnt:>4}")
