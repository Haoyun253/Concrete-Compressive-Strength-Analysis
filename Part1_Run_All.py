"""Execute the unchanged code cells from Part1_Concrete.ipynb. Run from this folder."""
from pathlib import Path
import sys
import importlib.metadata as meta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from IPython.display import display

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 160)
plt.rcParams.update({'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': 0.25})

print('Python:', sys.version.split()[0])
for name in ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'statsmodels', 'xlrd']:
    print(name, meta.version(name))

path = Path('Concrete_Data.xls')
if not path.exists():
    raise FileNotFoundError('Place Concrete_Data.xls beside this notebook.')

data = pd.read_excel(path, engine='xlrd')
names = ['cement', 'slag', 'ash', 'water', 'plasticizer', 'coarse', 'fine', 'age']
data.columns = names + ['strength']
assert data.shape == (1030, 9)
assert np.isfinite(data.to_numpy(dtype=float)).all()

test = data.iloc[500:630].copy()
train = pd.concat([data.iloc[:500], data.iloc[630:]]).copy()
assert len(train) == 900 and len(test) == 130
assert set(train.index).isdisjoint(test.index)
assert test.index.tolist() == list(range(500, 630))

x = train[names].to_numpy(dtype=float)
y = train['strength'].to_numpy(dtype=float)
xt = test[names].to_numpy(dtype=float)
yt = test['strength'].to_numpy(dtype=float)

scale = StandardScaler().fit(x)
xs = scale.transform(x)
xts = scale.transform(xt)
assert (x >= 0).all() and (xt >= 0).all()
xl = np.log1p(x)
xtl = np.log1p(xt)

print(f'Training: {len(train)} | Testing: {len(test)} | Missing: {data.isna().sum().sum()}')
display(train.head())
display(train.describe().loc[['mean', 'std', 'min', 'max']].round(3))

def design(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x])


def score(y, pred):
    mse = np.mean((y - pred) ** 2)
    r2 = 1 - mse / np.var(y, ddof=0)
    return float(mse), float(r2)


def gd(x, y, lr=None, steps=50000, tol=1e-10):
    a = design(x)
    n = len(y)
    if lr is None:
        L = np.linalg.eigvalsh(2 * a.T @ a / n)[-1]
        lr = 0.9 / L
    w = np.zeros(a.shape[1])
    loss = [np.mean((a @ w - y) ** 2)]
    stop = 'max steps'
    for i in range(steps):
        err = a @ w - y
        grad = 2 * a.T @ err / n
        w = w - lr * grad
        mse = np.mean((a @ w - y) ** 2)
        if not np.isfinite(mse):
            raise ValueError('Loss diverged. Try a smaller learning rate.')
        loss.append(float(mse))
        if abs(loss[-2] - loss[-1]) <= tol:
            stop = 'loss tolerance'
            break
    return {'w': w, 'loss': np.array(loss), 'lr': lr, 'stop': stop}


def evaluate(y, pred, yt, predt):
    mse, r2 = score(y, pred)
    mset, r2t = score(yt, predt)
    return {'Train MSE': mse, 'Test MSE': mset, 'Train R2': r2, 'Test R2': r2t}


def first_below(loss, value):
    ids = np.flatnonzero(loss < value)
    return int(ids[0]) if len(ids) else np.nan

out = Path('.')
labels = {'cement': 'Cement', 'slag': 'Blast furnace slag', 'ash': 'Fly ash',
          'water': 'Water', 'plasticizer': 'Superplasticizer',
          'coarse': 'Coarse aggregate', 'fine': 'Fine aggregate', 'age': 'Age'}


def save_table(table, file, title, note='', fmt='.6f'):
    text = table.copy()
    text.index = text.index.map(lambda s: labels.get(s, s))
    text = text.map(lambda v: format(v, fmt) if isinstance(v, (int, float)) else str(v))
    rows = [[str(i)] + row.tolist() for i, row in text.iterrows()]
    headers = [table.index.name or 'Model'] + list(table.columns)
    fig, ax = plt.subplots(figsize=(12, 1.7 + 0.45 * len(text)))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)
    tab = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.8)
    tab.auto_set_column_width(col=list(range(len(headers))))
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#d4dce5')
        if r == 0:
            cell.set_facecolor('#1f3b57')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#eef3f8' if r % 2 else 'white')
    fig.text(0.05, 0.03, note, fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out / file, dpi=220, bbox_inches='tight', facecolor='white')
    plt.show()


def curves(runs, title, file):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, run in runs.items():
        k = np.arange(len(run['loss']))
        axes[0].plot(k + 1, run['loss'], label=name)
        axes[1].plot(k, run['loss'], label=name)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Iteration + 1 (log scale)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_xlim(0, 500)
    for ax in axes:
        ax.axhline(np.var(y), color='gray', linestyle='--', label='Train R2 = 0')
        ax.set_ylabel('Training MSE (MPa squared)')
        ax.legend(fontsize=8)
    axes[0].set_title('Full training history')
    axes[1].set_title('First 500 iterations')
    fig.suptitle(title)
    fig.savefig(out / file, dpi=220, bbox_inches='tight', facecolor='white')
    plt.show()

g = {'Raw': gd(x, y), 'Standardized': gd(xs, y)}
rows = []
for name, a, at in [('Raw', x, xt), ('Standardized', xs, xts)]:
    run = g[name]
    row = {'Model': name, 'Learning rate': run['lr'],
           'Iterations': len(run['loss']) - 1,
           'First positive R2': first_below(run['loss'], np.var(y)),
           'Condition': np.linalg.cond(design(a)) ** 2,
           'Stop': run['stop']}
    row.update(evaluate(y, design(a) @ run['w'], yt, design(at) @ run['w']))
    rows.append(row)
gd_table = pd.DataFrame(rows).set_index('Model')
display(gd_table)
assert (gd_table['Train R2'] > 0).all()
curves(g, 'Q3.1. Raw vs. standardized features', 'Q3_1_Loss_Curves.png')

models = {'Raw': LinearRegression().fit(x, y),
          'Standardized': LinearRegression().fit(xs, y)}
rows = []
for name, a, at in [('Raw', x, xt), ('Standardized', xs, xts)]:
    model = models[name]
    rows.append({'Model': name, **evaluate(y, model.predict(a), yt, model.predict(at))})
display(pd.DataFrame(rows).set_index('Model').round(6))

raw = models['Raw']
scaled = models['Standardized']
m = scaled.coef_ / scale.scale_
b = scaled.intercept_ - m @ scale.mean_
display(pd.DataFrame({'Raw coefficient': np.r_[raw.intercept_, raw.coef_],
                      'Scaled, in raw units': np.r_[b, m]},
                     index=['intercept'] + names).round(8))
print('Maximum training prediction difference:',
      np.max(np.abs(raw.predict(x) - scaled.predict(xs))))
print('Maximum test prediction difference:',
      np.max(np.abs(raw.predict(xt) - scaled.predict(xts))))
assert np.allclose(raw.predict(x), scaled.predict(xs), rtol=1e-9, atol=1e-8)
assert np.allclose(raw.predict(xt), scaled.predict(xts), rtol=1e-9, atol=1e-8)

g['Log'] = gd(xl, y)
run = g['Log']
display(pd.DataFrame([{'Model': 'Log GD', 'Learning rate': run['lr'],
                       'Iterations': len(run['loss']) - 1, 'Stop': run['stop'],
                       **evaluate(y, design(xl) @ run['w'], yt, design(xtl) @ run['w'])}]))
curves(g, 'Q5.1. Raw, standardized, and log features', 'Q5_1_Loss_Curves.png')

models['Log'] = LinearRegression().fit(xl, y)
sets = {'Raw': (x, xt), 'Standardized': (xs, xts), 'Log': (xl, xtl)}
rows = []
for name, (a, at) in sets.items():
    model = models[name]
    rows.append({'Model': name, **evaluate(y, model.predict(a), yt, model.predict(at))})
summary = pd.DataFrame(rows).set_index('Model')

display(summary.round(6))
save_table(summary, 'Q5_3_Model_Comparison.png', 'Q5.3. Optimal multivariate models',
           'Unregularized least squares with an intercept. Training n = 900; testing n = 130. MSE in MPa squared.')

rows = []
for name, (a, at) in sets.items():
    run = g[name]
    best = summary.loc[name, 'Train MSE']
    target = np.var(y) - 0.95 * (np.var(y) - best)
    rows.append({'Model': name, 'Learning rate': run['lr'],
                 'Condition': np.linalg.cond(design(a)) ** 2,
                 'Iterations': len(run['loss']) - 1,
                 'First positive R2': first_below(run['loss'], np.var(y)),
                 'Steps to 95%': first_below(run['loss'], target),
                 'GD MSE': run['loss'][-1], 'Optimal MSE': best,
                 'MSE gap': run['loss'][-1] - best, 'Stop': run['stop']})
speed = pd.DataFrame(rows).set_index('Model')
display(speed)
print('NaN in Steps to 95% means the target was not reached within the run.')

uni = {}
fits = {}
rows = []
for j, name in enumerate(names):
    a, at = x[:, [j]], xt[:, [j]]
    run = gd(a, y)
    model = LinearRegression().fit(a, y)
    uni[name], fits[name] = run, model
    best, _ = score(y, model.predict(a))
    target = np.var(y) - 0.95 * (np.var(y) - best)
    rows.append({'Feature': name, 'Learning rate': run['lr'],
                 'Iterations': len(run['loss']) - 1,
                 'First positive R2': first_below(run['loss'], np.var(y)),
                 'Steps to 95%': first_below(run['loss'], target),
                 'MSE gap': run['loss'][-1] - best,
                 **evaluate(y, design(a) @ run['w'], yt, design(at) @ run['w'])})
uni_table = pd.DataFrame(rows).set_index('Feature')
display(uni_table)
count = int((uni_table['Train R2'] > 0).sum())
print('Univariate GD models with positive training R2:', count)
assert count >= 2

rows = []
for j, name in enumerate(names):
    model = fits[name]
    rows.append({'Feature': name, 'Univariate slope': model.coef_[0],
                 'Multivariate slope': raw.coef_[j],
                 'Univariate intercept': model.intercept_,
                 **evaluate(y, model.predict(x[:, [j]]), yt, model.predict(xt[:, [j]]))})
coef = pd.DataFrame(rows).set_index('Feature')

display(coef.round(6))
table = coef[['Univariate slope', 'Multivariate slope']].copy()
table.index.name = 'Feature'
save_table(table, 'Q6_2_Coefficients.png', 'Q6.2. Optimal coefficients in original input units',
           'All models include an intercept. Age: MPa/day; other slopes: MPa per kg/m3. Training n = 900.')

corr = train[names].corr()
print('Water / superplasticizer correlation:', corr.loc['water', 'plasticizer'])
print('Cement / fly ash correlation:', corr.loc['cement', 'ash'])

ols = {}
pvals = pd.DataFrame(index=names)
rows = []
for name, (a, at) in sets.items():
    fit = sm.OLS(y, design(a)).fit()
    ols[name] = fit
    pvals[name] = fit.pvalues[1:]
    rows.append({'Model': name, **evaluate(y, fit.predict(design(a)), yt, fit.predict(design(at)))})
    assert np.allclose(fit.predict(design(a)), models[name].predict(a), atol=1e-7)


pvals.index.name = 'Feature'
display(pvals.style.format('{:.3e}'))
save_table(pvals, 'Q7_1_P_Values.png', 'Q7.1. Feature p-values in three multivariate models',
           'Classical OLS, two-sided tests, intercept included but not shown. Training n = 900.', fmt='.3e')
display(pd.DataFrame(rows).set_index('Model').round(6))
assert np.allclose(pvals['Raw'], pvals['Standardized'], atol=1e-10, rtol=1e-6)