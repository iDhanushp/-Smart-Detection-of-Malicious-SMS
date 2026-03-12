"""
generate_spam_samples.py
========================
Generates ~3000 synthetic Indian promotional SMS samples across 25 categories.
Output: datasetgenerateor/spam_synthetic_3000.csv  (address, body, label=spam, category)

Run:
    python generate_spam_samples.py
"""

import csv
import os
import random
import string

random.seed(42)

BASE_DIR = os.path.dirname(__file__)
OUT_PATH = os.path.join(BASE_DIR, 'spam_synthetic_3000.csv')
MASTER_PATH = os.path.join(BASE_DIR, 'spam_master.csv')


def rphone():
    pfx = random.choice([
        '70', '72', '73', '74', '75', '76', '77', '78', '79',
        '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
        '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'
    ])
    return '+91' + pfx + ''.join(random.choices(string.digits, k=8))


def dlt_sender(*brands):
    brand = random.choice(brands)
    prefix = random.choice(['AD', 'AX', 'BZ', 'CP', 'DM', 'HP', 'JD', 'JM', 'TX', 'VK', 'VM'])
    return f'{prefix}-{brand[:6].upper()}'


def ramount(lo=50, hi=10000, step=50):
    return random.randrange(lo, hi + step, step)


def rpercent(lo=5, hi=80):
    return random.randint(lo, hi)


def rdate():
    return f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2026"


def rtime():
    return f"{random.randint(8, 22):02d}:{random.choice(['00', '10', '15', '20', '30', '45'])}"


def rcode(n=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def rshort(domains=None):
    if domains is None:
        domains = ['bit.ly', 'tinyurl.com', 'rb.gy', 'cutt.ly', 't.ly', 'lihi.cc', 'short.gy', 's.id']
    dom = random.choice(domains)
    path = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 9)))
    return f'https://{dom}/{path}'


def rbrand_url(*domains):
    dom = random.choice(domains)
    path = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(4, 10)))
    return f'https://{dom}/{path}'


rows = []
seen_bodies = set()


def add(category, sender, body):
    body = ' '.join(str(body).split())
    if body in seen_bodies:
        return False
    seen_bodies.add(body)
    rows.append((sender, body, 'spam', category))
    return True


GAMING_APPS = ['Junglee Rummy', 'A23 Rummy', 'RummyCircle', 'RummyTime', 'PokerBaazi']
FANTASY_APPS = ['Dream11', 'My11Circle', 'Gamezy', 'MPL Fantasy', 'Vision11']
CASINO_APPS = ['JeetWin', 'PariPlay', 'MegaSpin', 'Royal Ace', 'WinBuzz']
WALLETS = ['Paytm', 'PhonePe', 'Amazon Pay', 'Mobikwik', 'Freecharge']
FOOD_APPS = ['Swiggy', 'Zomato', 'EatSure', 'Magicpin', 'Dominos']
ECOM_APPS = ['Amazon', 'Flipkart', 'Myntra', 'Meesho', 'Nykaa']
LOAN_APPS = ['KreditBee', 'MoneyView', 'PaySense', 'Fibe', 'Navi']
CARD_BRANDS = ['HDFC Bank', 'ICICI Bank', 'Axis Bank', 'SBI Card', 'RBL Bank']
BNPL_APPS = ['LazyPay', 'Simpl', 'ZestMoney', 'Amazon Pay Later', 'Flipkart Pay Later']
INSURANCE = ['PolicyBazaar', 'Acko', 'Digit', 'ICICI Lombard', 'Star Health']
TRAVEL = ['MakeMyTrip', 'Goibibo', 'Yatra', 'Cleartrip', 'Ixigo']
MOVIES = ['BookMyShow', 'PVR', 'INOX', 'Paytm Movies', 'Miraj']
TELCOS = ['Airtel', 'Jio', 'Vi', 'BSNL']
APP_STORES = ['Google Play', 'Galaxy Store', 'App Bazaar', 'TapToInstall']
BEAUTY = ['Nykaa', 'Purplle', 'Mamaearth', 'MyGlamm', 'Tira']
GROCERY = ['Blinkit', 'Zepto', 'BigBasket', 'JioMart', 'Instamart']
EDTECH = ['UpGrad', 'Unacademy', 'BYJUS', 'PhysicsWallah', 'SkillUp']
REALTY = ['NoBroker', 'MagicBricks', 'Housing', 'SquareYards', '99acres']
AUTO = ['Cars24', 'Spinny', 'GoMechanic', 'Pitstop', 'CarDekho']
WELLNESS = ['Cult', 'Tata 1mg', 'HealthKart', 'PharmEasy', 'FitPass']
EVENTS = ['BookMyShow Live', 'District', 'Insider', 'SkillFest', 'CampusBuzz']


CATEGORY_COUNTS = {
    'rummy_bonus': 220,
    'fantasy_sports': 180,
    'casino_vip': 140,
    'poker_tourney': 120,
    'cashback_wallet': 180,
    'food_delivery_offer': 160,
    'ecommerce_sale': 180,
    'instant_loan_offer': 150,
    'credit_card_offer': 110,
    'bnpl_offer': 110,
    'insurance_marketing': 90,
    'travel_flash_sale': 110,
    'movie_ticket_offer': 90,
    'telecom_recharge': 120,
    'data_pack_offer': 100,
    'app_install_campaign': 100,
    'referral_program': 90,
    'festival_sale': 120,
    'beauty_fashion_sale': 100,
    'grocery_offer': 100,
    'edtech_promo': 90,
    'real_estate_lead': 90,
    'auto_service_offer': 80,
    'wellness_offer': 80,
    'local_event_promo': 90,
}

assert sum(CATEGORY_COUNTS.values()) == 3000


def generate_until(category, target, builder):
    created = 0
    attempts = 0
    while created < target:
        attempts += 1
        sender, body = builder()
        if add(category, sender, body):
            created += 1
        if attempts > target * 20:
            raise RuntimeError(f'Could not generate enough unique rows for {category}')


def build_rummy_bonus():
    app = random.choice(GAMING_APPS)
    amount = ramount(200, 12000, 50)
    bonus = ramount(50, 3000, 50)
    url = random.choice([rshort(), rbrand_url('gmg.im', 'rmy.onl', 'playnow.app')])
    sender = random.choice([dlt_sender('JERUMY', 'JLRUMY', 'RUMMYX'), rphone()])
    body = random.choice([
        f"{app}: Free Rs.{bonus} welcome bonus + up to Rs.{amount} cash table. Play now {url}",
        f"Congrats! Your {app} account unlocked Rs.{amount} instant cash games. Join now {url}",
        f"{app} offer: deposit Rs.{bonus} get Rs.{amount} playable chips today only {url}",
        f"Weekend rummy dhamaka on {app}. Win up to Rs.{amount} cash. Register now {url}",
        f"Get Rs.{bonus} joining bonus on {app} and enter cash pool of Rs.{amount}. {url}",
    ])
    return sender, body


def build_fantasy_sports():
    app = random.choice(FANTASY_APPS)
    amount = ramount(1000, 200000, 500)
    contest = ramount(49, 999, 10)
    url = random.choice([rshort(), rbrand_url('fantasy11.app', 'leaguezone.in', 'play11.win')])
    sender = dlt_sender('DREAM', 'MY11CL', 'GAMEZY', 'FANTSY')
    body = random.choice([
        f"{app}: Join mega contest from Rs.{contest} and win up to Rs.{amount}. Create team now {url}",
        f"Tonight's match on {app} has Rs.{amount} prize pool. Entry starts Rs.{contest}. {url}",
        f"Only today: extra cashback on {app} fantasy leagues. Play and win Rs.{amount}. {url}",
        f"Your fantasy wallet is eligible for booster contest on {app}. Join now {url}",
    ])
    return sender, body


def build_casino_vip():
    app = random.choice(CASINO_APPS)
    amount = ramount(1000, 50000, 100)
    url = random.choice([rshort(), rbrand_url('vipspin.live', 'casinozone.bet', 'playace.in')])
    sender = random.choice([dlt_sender('CASINO', 'MEGASP', 'WINVIP'), rphone()])
    body = random.choice([
        f"{app}: VIP tables open now. Grab Rs.{amount} bonus chips and spin instantly {url}",
        f"Play roulette, slots and live casino on {app}. Bonus up to Rs.{amount}. {url}",
        f"Exclusive {app} VIP pass for you. Deposit today and unlock Rs.{amount} reward {url}",
    ])
    return sender, body


def build_poker_tourney():
    app = random.choice(['PokerBaazi', 'Adda52', 'Spartan Poker', 'PokerSaint'])
    amount = ramount(5000, 300000, 500)
    buyin = ramount(99, 1999, 50)
    url = random.choice([rshort(), rbrand_url('pokerhub.in', 'tourneys.live', 'tables.app')])
    sender = dlt_sender('POKER', 'TOURNY', 'ADDA52')
    body = random.choice([
        f"{app}: Daily poker tournament live now. Buy-in Rs.{buyin}, prize pool Rs.{amount}. {url}",
        f"Register on {app} and claim seat in today's Rs.{amount} GTD event. {url}",
        f"Final call: {app} tournament starts at {rtime()}. Join from Rs.{buyin}. {url}",
    ])
    return sender, body


def build_cashback_wallet():
    wallet = random.choice(WALLETS)
    percent = rpercent(10, 70)
    amount = ramount(50, 2000, 50)
    url = random.choice([rbrand_url('paytm.me', 'phone.pe', 'freecharge.in', 'offers.wallet')])
    sender = dlt_sender('PAYTM', 'PHONEP', 'CASHBK', 'OFFERS')
    body = random.choice([
        f"{wallet}: Get {percent}% cashback up to Rs.{amount} on recharge, bill pay & scan and pay. Use now {url}",
        f"Limited offer from {wallet}. Flat Rs.{amount} cashback on first UPI payment today {url}",
        f"Pay via {wallet} before {rtime()} and grab extra {percent}% cashback. T&C apply {url}",
        f"Your {wallet} promo is active now. Shop, recharge or pay bills to earn Rs.{amount}. {url}",
    ])
    return sender, body


def build_food_delivery_offer():
    app = random.choice(FOOD_APPS)
    percent = rpercent(40, 80)
    amount = ramount(75, 400, 25)
    url = random.choice([rbrand_url('swiggy.in', 'zoma.to', 'eatsure.in', 'foodoffers.app')])
    sender = dlt_sender('SWIGGY', 'ZOMATO', 'FOODIE', 'OFFERS')
    body = random.choice([
        f"{app}: Get up to {percent}% OFF + free delivery on your next order. Save up to Rs.{amount}. {url}",
        f"Hungry? {app} weekend deal unlocked. Flat Rs.{amount} off on select restaurants {url}",
        f"Order now on {app} and enjoy {percent}% discount before {rtime()}. {url}",
        f"Your {app} coupon is live. Use code SAVE{random.randint(50, 500)} and save Rs.{amount}. {url}",
    ])
    return sender, body


def build_ecommerce_sale():
    app = random.choice(ECOM_APPS)
    percent = rpercent(20, 80)
    amount = ramount(100, 5000, 100)
    url = random.choice([rbrand_url('amzn.to', 'fkrt.it', 'myntra.com', 'meesho.io', 'nykaa.com')])
    sender = dlt_sender('AMZON', 'FLPKRT', 'MYNTRA', 'MESHOO', 'NYKAAA')
    body = random.choice([
        f"{app} Sale Live: Up to {percent}% off on top picks. Extra Rs.{amount} off today {url}",
        f"Hurry! {app} flash sale starts now. Claim coupon worth Rs.{amount} before stock ends {url}",
        f"Only for you: exclusive {app} offer up to {percent}% OFF + prepaid savings. {url}",
        f"Wishlist items on {app} are now on sale. Save Rs.{amount} instantly {url}",
    ])
    return sender, body


def build_instant_loan_offer():
    app = random.choice(LOAN_APPS)
    amount = ramount(10000, 500000, 1000)
    rate = round(random.uniform(1.0, 2.5), 2)
    url = random.choice([rshort(), rbrand_url('loanfast.in', 'cashnow.app', 'applyloan.co')])
    sender = random.choice([dlt_sender('KREDIT', 'MONEYV', 'LOANOK', 'FIBELO'), rphone()])
    body = random.choice([
        f"{app}: Instant personal loan up to Rs.{amount} at {rate}% monthly. Apply in 2 min {url}",
        f"Pre-approved offer from {app}. Get up to Rs.{amount} today with minimal docs {url}",
        f"Need cash? {app} can disburse Rs.{amount} directly to your bank. Apply now {url}",
        f"Limited period: {app} reduces processing fee on loans up to Rs.{amount}. {url}",
    ])
    return sender, body


def build_credit_card_offer():
    bank = random.choice(CARD_BRANDS)
    cashback = rpercent(5, 25)
    points = ramount(500, 10000, 100)
    url = random.choice([rbrand_url('cardsales.in', 'applycard.app', 'cardperks.co')])
    sender = dlt_sender('HDFCBK', 'ICICIB', 'AXISBK', 'SBICRD', 'RBLBNK')
    body = random.choice([
        f"{bank}: Upgrade to a premium card and get {cashback}% cashback + {points} bonus points. Apply {url}",
        f"Exclusive card offer from {bank}. Zero joining fee and rewards worth Rs.{points}. {url}",
        f"You're eligible for a new {bank} credit card with airport lounge benefits. Apply now {url}",
    ])
    return sender, body


def build_bnpl_offer():
    app = random.choice(BNPL_APPS)
    limit_amt = ramount(2000, 100000, 500)
    url = random.choice([rbrand_url('paylater.in', 'bnpl.app', 'shopcredit.co')])
    sender = dlt_sender('LAZPAY', 'SIMPLX', 'BNPLOK', 'ZESTMY')
    body = random.choice([
        f"{app}: Shop now and pay later. Credit line up to Rs.{limit_amt} activated for you {url}",
        f"Your {app} pay-later limit is ready: Rs.{limit_amt}. Use on shopping, travel and food {url}",
        f"Split bills with {app}. No-cost EMI and pay-later limit up to Rs.{limit_amt}. {url}",
    ])
    return sender, body


def build_insurance_marketing():
    brand = random.choice(INSURANCE)
    amount = ramount(200000, 5000000, 100000)
    url = random.choice([rbrand_url('policybzr.com', 'acko.co', 'digit.insure', 'plans.health')])
    sender = dlt_sender('POLICY', 'ACKO', 'DIGIT', 'HLTHPL')
    body = random.choice([
        f"{brand}: Health cover up to Rs.{amount} with cashless hospitals. Compare plans now {url}",
        f"Renew or buy with {brand} and save more on your policy premium today {url}",
        f"Family health plans from {brand} now start at affordable monthly premiums. {url}",
    ])
    return sender, body


def build_travel_flash_sale():
    brand = random.choice(TRAVEL)
    percent = rpercent(15, 55)
    amount = ramount(500, 8000, 100)
    url = random.choice([rbrand_url('mmt.to', 'goibibo.in', 'yatraa.in', 'tripoffer.co')])
    sender = dlt_sender('MAKEMY', 'GOIBIB', 'YATRAA', 'TRAVEL')
    body = random.choice([
        f"{brand}: Flash sale live. Save up to Rs.{amount} on flights & hotels. Book now {url}",
        f"Travel with {brand} and enjoy up to {percent}% off on summer bookings. {url}",
        f"Weekend getaway alert from {brand}. Extra Rs.{amount} off with limited seats. {url}",
    ])
    return sender, body


def build_movie_ticket_offer():
    brand = random.choice(MOVIES)
    amount = ramount(100, 500, 50)
    url = random.choice([rbrand_url('pvr.im', 'bookmy.show', 'movieoffer.in')])
    sender = dlt_sender('BOOKMY', 'PVRCIN', 'INOXMX', 'MOVIES')
    body = random.choice([
        f"{brand}: Get 1+1 movie tickets or save Rs.{amount} on weekend bookings. Grab now {url}",
        f"Tonight only on {brand}: popcorn combo + ticket discounts. Save Rs.{amount}. {url}",
        f"Your movie pass is waiting. Book on {brand} and unlock flat Rs.{amount} off {url}",
    ])
    return sender, body


def build_telecom_recharge():
    telco = random.choice(TELCOS)
    amount = ramount(19, 399, 10)
    data = random.choice(['1GB/day', '2GB/day', '3GB/day', 'unlimited 5G'])
    url = random.choice([rbrand_url('airtel.in', 'jio.com', 'vi.in', 'recharge-now.app')])
    sender = dlt_sender('AIRTEL', 'JIOINF', 'VODAFO', 'BSNLIN')
    body = random.choice([
        f"{telco}: Recharge with Rs.{amount} and get {data} + OTT benefits today {url}",
        f"Special {telco} recharge pack live now. Save on prepaid plan before {rtime()} {url}",
        f"Grab your {telco} booster pack of {data}. Recharge at just Rs.{amount}. {url}",
    ])
    return sender, body


def build_data_pack_offer():
    telco = random.choice(TELCOS)
    amount = ramount(9, 99, 10)
    gb = random.choice([1, 2, 3, 5, 10])
    url = random.choice([rbrand_url('airtel.in', 'jio.com', 'vi.in', 'datadeal.app')])
    sender = dlt_sender('DATAPK', 'JIODAT', 'AIRTEL', 'VIOFER')
    body = random.choice([
        f"{telco}: Need more data? Get {gb}GB booster at Rs.{amount} only. Activate now {url}",
        f"Your {telco} data booster is available. Buy {gb}GB for Rs.{amount} before midnight {url}",
        f"Flat Rs.{amount} for {gb}GB high-speed data on {telco}. Tap to recharge {url}",
    ])
    return sender, body


def build_app_install_campaign():
    store = random.choice(APP_STORES)
    reward = ramount(20, 500, 10)
    url = random.choice([rshort(), rbrand_url('play.google.com', 'appbonus.in', 'installnow.app')])
    sender = random.choice([dlt_sender('APPBON', 'INSTAL', 'GETAPP'), rphone()])
    body = random.choice([
        f"Install the featured app from {store} and earn reward up to Rs.{reward}. Start now {url}",
        f"New app campaign live. Download today and unlock Rs.{reward} wallet credit {url}",
        f"Tap to install and collect bonus up to Rs.{reward}. Limited devices only {url}",
    ])
    return sender, body


def build_referral_program():
    brand = random.choice(WALLETS + FOOD_APPS + ECOM_APPS)
    reward = ramount(50, 1000, 25)
    code = rcode(6)
    url = random.choice([rbrand_url('refer.app', 'invite-now.in', 'earnbonus.co')])
    sender = dlt_sender('REFRAL', 'INVITE', 'EARNBK')
    body = random.choice([
        f"Invite friends to {brand} with code {code} and earn Rs.{reward} for every signup {url}",
        f"Your {brand} referral campaign is live. Share code {code} and get Rs.{reward}. {url}",
        f"Refer & earn on {brand}: flat Rs.{reward} per successful install. Start now {url}",
    ])
    return sender, body


def build_festival_sale():
    brand = random.choice(ECOM_APPS + FOOD_APPS + WALLETS)
    percent = rpercent(25, 80)
    amount = ramount(100, 3000, 50)
    url = random.choice([rshort(), rbrand_url('festive-offers.in', 'salezone.app', 'savebig.co')])
    sender = dlt_sender('FESTIV', 'MEGASL', 'BIGDAY')
    body = random.choice([
        f"Festive sale on {brand}: up to {percent}% OFF + extra Rs.{amount} off today only {url}",
        f"Celebrate with {brand}. Limited festival deal unlocked for your number. Shop now {url}",
        f"Mega savings are live on {brand}. Use your festive coupon before stock ends {url}",
    ])
    return sender, body


def build_beauty_fashion_sale():
    brand = random.choice(BEAUTY)
    percent = rpercent(20, 75)
    amount = ramount(150, 2500, 50)
    url = random.choice([rbrand_url('nykaa.com', 'purplle.com', 'tira.co', 'beautydeals.app')])
    sender = dlt_sender('BEAUTY', 'NYKAAA', 'FASHON', 'GLAMUP')
    body = random.choice([
        f"{brand}: Beauty & fashion sale live. Save up to {percent}% + Rs.{amount} extra off {url}",
        f"Glow up with {brand}. Grab lipstick, skincare and makeup deals before {rtime()} {url}",
        f"Your {brand} coupon is active. Unlock flat Rs.{amount} off on top products {url}",
    ])
    return sender, body


def build_grocery_offer():
    brand = random.choice(GROCERY)
    amount = ramount(75, 500, 25)
    mins = random.choice([10, 15, 20, 30])
    url = random.choice([rbrand_url('blinkit.com', 'zepto.app', 'jiomart.com', 'grocery-now.in')])
    sender = dlt_sender('BLINKT', 'ZEPTOO', 'BBASKT', 'JIOMRT')
    body = random.choice([
        f"{brand}: Save Rs.{amount} on grocery orders and get delivery in {mins} min. Order now {url}",
        f"Daily essentials deal from {brand}. Flat Rs.{amount} off on orders above minimum cart {url}",
        f"Quick grocery savings with {brand}. Extra coupon active till {rtime()} {url}",
    ])
    return sender, body


def build_edtech_promo():
    brand = random.choice(EDTECH)
    percent = rpercent(20, 70)
    amount = ramount(1000, 25000, 500)
    url = random.choice([rbrand_url('learn-now.in', 'upgrad.co', 'studyoffer.app')])
    sender = dlt_sender('EDTECH', 'UPGRAD', 'STUDYX', 'COURSE')
    body = random.choice([
        f"{brand}: Upgrade your career with up to {percent}% scholarship on premium courses. Apply {url}",
        f"New learner offer from {brand}. Save Rs.{amount} on job-ready programs today {url}",
        f"Admissions open on {brand}. Limited seats, expert mentors, easy EMI. Enrol now {url}",
    ])
    return sender, body


def build_real_estate_lead():
    brand = random.choice(REALTY)
    amount = ramount(50000, 5000000, 50000)
    city = random.choice(['Bengaluru', 'Mumbai', 'Pune', 'Hyderabad', 'Chennai'])
    url = random.choice([rbrand_url('homesale.in', 'realtydeal.co', 'propertylead.app')])
    sender = random.choice([dlt_sender('NOBRKR', 'HOUSES', 'REALTY'), rphone()])
    body = random.choice([
        f"{brand}: New homes in {city} from Rs.{amount}. Site visit offers available {url}",
        f"Looking for property in {city}? {brand} has verified listings and exclusive builder deals {url}",
        f"Book your apartment visit with {brand} and unlock launch pricing in {city}. {url}",
    ])
    return sender, body


def build_auto_service_offer():
    brand = random.choice(AUTO)
    amount = ramount(200, 3000, 100)
    url = random.choice([rbrand_url('carcare.in', 'autodeal.app', 'servicecar.co')])
    sender = dlt_sender('CARSRV', 'AUTOFX', 'GARAGE')
    body = random.choice([
        f"{brand}: Car service offer unlocked. Save Rs.{amount} on pickup & drop maintenance {url}",
        f"Need a used car or service? {brand} has special deals waiting for you {url}",
        f"Book with {brand} today and get flat Rs.{amount} off on your next service {url}",
    ])
    return sender, body


def build_wellness_offer():
    brand = random.choice(WELLNESS)
    percent = rpercent(15, 60)
    amount = ramount(150, 2500, 50)
    url = random.choice([rbrand_url('healthdeal.in', 'fitpass.app', 'medsave.co')])
    sender = dlt_sender('HEALTH', 'FITPAS', 'WELLNS')
    body = random.choice([
        f"{brand}: Save up to {percent}% on wellness plans, lab tests and supplements today {url}",
        f"Your {brand} health coupon is ready. Get flat Rs.{amount} off before expiry {url}",
        f"Fitness & health offers from {brand} are live now. Tap to claim benefits {url}",
    ])
    return sender, body


def build_local_event_promo():
    brand = random.choice(EVENTS)
    city = random.choice(['Bengaluru', 'Mumbai', 'Pune', 'Hyderabad', 'Delhi'])
    amount = ramount(100, 1500, 50)
    url = random.choice([rbrand_url('eventsnow.in', 'bookmy.show', 'district.in', 'livepass.app')])
    sender = dlt_sender('EVENTS', 'TICKTS', 'CAMPUS', 'LIVESH')
    body = random.choice([
        f"{brand}: Live events in {city} this weekend. Tickets from Rs.{amount}. Book now {url}",
        f"Explore comedy, music and workshops on {brand}. Early-bird prices active {url}",
        f"Your city pass from {brand} unlocks discounts on shows in {city}. Grab it now {url}",
    ])
    return sender, body


BUILDERS = {
    'rummy_bonus': build_rummy_bonus,
    'fantasy_sports': build_fantasy_sports,
    'casino_vip': build_casino_vip,
    'poker_tourney': build_poker_tourney,
    'cashback_wallet': build_cashback_wallet,
    'food_delivery_offer': build_food_delivery_offer,
    'ecommerce_sale': build_ecommerce_sale,
    'instant_loan_offer': build_instant_loan_offer,
    'credit_card_offer': build_credit_card_offer,
    'bnpl_offer': build_bnpl_offer,
    'insurance_marketing': build_insurance_marketing,
    'travel_flash_sale': build_travel_flash_sale,
    'movie_ticket_offer': build_movie_ticket_offer,
    'telecom_recharge': build_telecom_recharge,
    'data_pack_offer': build_data_pack_offer,
    'app_install_campaign': build_app_install_campaign,
    'referral_program': build_referral_program,
    'festival_sale': build_festival_sale,
    'beauty_fashion_sale': build_beauty_fashion_sale,
    'grocery_offer': build_grocery_offer,
    'edtech_promo': build_edtech_promo,
    'real_estate_lead': build_real_estate_lead,
    'auto_service_offer': build_auto_service_offer,
    'wellness_offer': build_wellness_offer,
    'local_event_promo': build_local_event_promo,
}


for category, count in CATEGORY_COUNTS.items():
    generate_until(category, count, BUILDERS[category])

random.shuffle(rows)

with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['address', 'body', 'label', 'category'])
    writer.writerows(rows)

with open(MASTER_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['address', 'body', 'label'])
    for sender, body, label, _category in rows:
        writer.writerow([sender, body, label])

print(f'Generated {len(rows):,} synthetic SPAM rows')
print(f'Wrote: {OUT_PATH}')
print(f'Wrote: {MASTER_PATH}')
print('Category counts:')
for category, count in CATEGORY_COUNTS.items():
    print(f'  {category:<22} {count:>4}')
