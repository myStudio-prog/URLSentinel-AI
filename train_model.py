"""URLSentinel AI: reproducible URL-only phishing baseline.
Run: python train_model.py
"""
import json
import re
import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
OUT = Path('artifacts')
OUT.mkdir(exist_ok=True)
FEATURES = [
    'url_length','domain_length','is_ip','dot_count','hyphen_count',
    'digit_count','letter_count','special_char_count','at_count',
    'question_count','equals_count','uses_https'
]
MD_LINK = re.compile(r'^\[([^\]]+)\]\((https?://[^\s)]+)\)$', re.IGNORECASE)


def clean_url(raw):
    value = str(raw).strip()
    match = MD_LINK.fullmatch(value)
    if match:
        value = match.group(2)
    if any(ch.isspace() for ch in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port  # validates port syntax
    except ValueError:
        return None
    if parsed.scheme.lower() not in {'http', 'https'} or not host:
        return None
    return value


def extract_features(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    return {
        'url_length': len(url),
        'domain_length': len(host),
        'is_ip': is_ip,
        'dot_count': url.count('.'),
        'hyphen_count': url.count('-'),
        'digit_count': sum(ch.isdigit() for ch in url),
        'letter_count': sum(ch.isalpha() for ch in url),
        'special_char_count': sum(not ch.isalnum() for ch in url),
        'at_count': url.count('@'),
        'question_count': url.count('?'),
        'equals_count': url.count('='),
        'uses_https': int(parsed.scheme.lower() == 'https'),
    }


def metrics(y_true, pred):
    return {
        'accuracy': accuracy_score(y_true, pred),
        'phishing_precision': precision_score(y_true, pred, pos_label=0, zero_division=0),
        'phishing_recall': recall_score(y_true, pred, pos_label=0, zero_division=0),
        'phishing_f1': f1_score(y_true, pred, pos_label=0, zero_division=0),
        'confusion_matrix_labels_0_1': confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def train_model():
    ds = fetch_ucirepo(id=967)
    source = ds.data.features
    targets = ds.data.targets
    frame = pd.DataFrame({'url': source['URL'].map(clean_url), 'label': targets['label'].to_numpy()})
    original_count = len(frame)
    frame = frame.dropna(subset=['url'])
    frame = frame[frame['label'].isin([0, 1])].copy()
    frame['url_key'] = frame['url'].str.lower()
    conflicting = frame.groupby('url_key')['label'].nunique()
    bad = set(conflicting[conflicting > 1].index)
    frame = frame[~frame['url_key'].isin(bad)].drop_duplicates('url_key').reset_index(drop=True)
    print('Original:', original_count, '| Cleaned unique:', len(frame), '| Conflicting URL groups:', len(bad))
    print('Labels (0 phishing, 1 legitimate):', frame['label'].value_counts().to_dict())
    x = pd.DataFrame([extract_features(u) for u in frame['url']], columns=FEATURES)
    y = frame['label']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = {'dataset_id': 967, 'original_records': original_count,
               'cleaned_records': len(frame), 'train_records': len(x_train),
               'test_records': len(x_test), 'feature_names': FEATURES,
               'random_split': metrics(y_test, predictions),
               'limitations': 'Random split of one dataset; not independent real-world performance.'}
    print('\nRANDOM SPLIT RESULTS:')
    print(json.dumps(results['random_split'], indent=2))

    # Separate, stricter evaluation using disjoint exact hostnames.
    groups = frame['url'].map(lambda u: urlsplit(u).hostname or '')
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(splitter.split(x, y, groups=groups))
    group_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=SEED, n_jobs=-1)
    group_model.fit(x.iloc[tr], y.iloc[tr])
    results['hostname_split'] = metrics(y.iloc[te], group_model.predict(x.iloc[te]))
    results['hostname_split_records'] = {'train': len(tr), 'test': len(te)}
    print('\nHOSTNAME-SEPARATED RESULTS:')
    print(json.dumps(results['hostname_split'], indent=2))

    joblib.dump({'model': model, 'features': FEATURES, 'phishing_label': 0,
                 'version': 'urlsentinel-v1'}, OUT / 'urlsentinel_model.pkl')
    (OUT / 'evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nSAVED:', OUT / 'urlsentinel_model.pkl', 'and', OUT / 'evaluation.json')

if __name__ == '__main__':
    train_model()
