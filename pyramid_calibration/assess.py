"""Make the prespecified MSE-based decision before launching generation."""
import math
from .common import *

def main():
    assert read(ROOT/'DENOISING_COMPLETE.json')['complete']
    measurements=[read(STORE/'denoising'/f'{key(t)}.json')for t in TEST_TIMES]
    for d in measurements:
        assert d['complete'] and all(math.isfinite(v)and v>=0 for v in d['mse'].values())
    low=[d for d in measurements if d['t']>=.8]
    improved=[d for d in low if d['paired_improvement']['conditional_calibrated']['mean']>0]
    significant_harm=[d['t']for d in low if d['paired_improvement']['conditional_calibrated']['mean']<
                      -3*d['paired_improvement']['conditional_calibrated']['standard_error']]
    if not improved or significant_harm:
        save(ROOT/'GENERATION_REVIEW_NEEDED.json',{'reason':'Review low-noise held-out MSE before choosing a correction schedule',
            'low_noise_improved_times':[d['t']for d in improved],'significant_harm_times':significant_harm})
        raise RuntimeError('Denoising assessment needs review; generation has not been authorized by this gate.')
    assessment=('Held-out raw conditional denoising improves at '+', '.join(str(d['t'])for d in improved)+
                '. There is no >3-SE MSE deterioration at tested t>=0.8. Retain the full and JiT-style half-late '
                'schedules for generation, with a paired uncalibrated control. This decision uses denoising only; '
                'generation FID has not been used to choose coefficients or schedules.')
    save(ROOT/'GENERATION_DECISION.json',{'approved_variants':{'baseline':True,'calibrated':True,'late_half':True},
        'criterion':'finite results; positive conditional low-noise MSE gain at one or more tested times; '
                    'no >3-SE conditional low-noise harm; otherwise review before generation',
        'assessment':assessment,'denoising_sha256':{key(d['t']):digest(STORE/'denoising'/f"{key(d['t'])}.json")for d in measurements}})
    from .report import render_workspace
    render_workspace(ROOT)

if __name__=='__main__':main()
