"""제출물 패키징 — 소스코드 zip 구성 + 테스트데이터 예측결과 파일 생성.

구성(요강):
  - 소스코드 + requirements.txt
  - 학습용 데이터
  - README
  - 테스트데이터 예측결과 파일
zip 자체는 커밋하지 않는다(.gitignore). 원시 데이터는 저장소에 올리지 않고 zip 에만 넣는다.
"""
import sys, zipfile, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final_baseline"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
OUT = ROOT / "submission"; OUT.mkdir(exist_ok=True)
DATA = ROOT / "5. 자원 최적화 AI 데이터셋" / "okm_augumented_2021.csv"


def make_predictions():
    """배포 아티팩트로 예측결과 파일을 만든다. 학습에 쓰이지 않은 마지막 구간만 대상."""
    import features as FT, inference as IN
    q, mode = FT.build()
    d = FT.common_rows(q, mode)
    # 공식 테스트셋이 별도 제공되지 않으므로, 평가에 쓴 마지막 fold 구간(TEST)을 대상으로 한다.
    from sklearn.model_selection import TimeSeriesSplit
    import yaml
    cfg = yaml.safe_load((ROOT / "analysis/final_baseline/config.yaml").read_text(encoding="utf-8"))
    folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
    te = d.iloc[folds[-1][1]]
    out = IN.predict(te, IN.load())
    for h in (1, 2, 3, 4):
        out[f"actual_{h*15}m"] = te[f"y_h{h}"].values
    p = OUT / "test_predictions.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[pred] {p.relative_to(ROOT)}  rows={len(out)}  "
          f"기간 {out.timestamp.min()} ~ {out.timestamp.max()}")
    return p


def make_zip():
    z = OUT / "소스코드_전력피크조기경보.zip"
    if z.exists(): z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for pat in ("analysis/**/*.py", "analysis/**/*.yaml", "analysis/reports/*.md",
                    "analysis/state/*.csv", "analysis/state/*.md"):
            for p in ROOT.glob(pat):
                if "artifacts" in p.parts: continue
                f.write(p, str(p.relative_to(ROOT)))
        for p in (ROOT / "requirements.txt", ROOT / "README.md", ROOT / "ENVIRONMENT.txt"):
            if p.exists(): f.write(p, p.name)
        if DATA.exists():
            f.write(DATA, f"data/{DATA.name}")      # 학습용 데이터
        pr = OUT / "test_predictions.csv"
        if pr.exists(): f.write(pr, "test_predictions.csv")
        for p in sorted((ROOT / "analysis/tables").glob("*.csv")):
            if p.stat().st_size < 2_000_000:
                f.write(p, f"analysis/tables/{p.name}")
        for p in sorted((ROOT / "analysis/figures").glob("*.png")):
            f.write(p, f"analysis/figures/{p.name}")
    print(f"[zip ] {z.relative_to(ROOT)}  {z.stat().st_size/1e6:.1f} MB")
    return z


if __name__ == "__main__":
    make_predictions()
    make_zip()
