"""Provide test for parcel aggregation."""

# Authors: Amir Omidvarnia <a.omidvarnia@fz-juelich.de>
#          Kaustubh R. Patil <k.patil@fz-juelich.de>
# License: AGPL

from nilearn import datasets, image
from junifer.markers.alff_falff_atlas \
    import AmplitudeLowFrequencyFluctuationAtlas


def test_AmplitudeLowFrequencyFluctuationAtlas() -> None:
    """Test AmplitudeLowFrequencyFluctuationAtlas."""

    # get a dataset
    ni_data = datasets.fetch_spm_auditory(subject_id='sub001')
    fmri_img = image.concat_imgs(ni_data.func)  # type: ignore
    falff = AmplitudeLowFrequencyFluctuationAtlas(atlas='Schaefer100x7',
                                                  TR=.72)
    out = falff.compute({'data': fmri_img})

    assert 'ALFF' in out
    assert 'fALFF' in out
    assert 'roi_names' in out
    assert out['ALFF'].shape[0] == 100
    assert out['fALFF'].shape[0] == 100
    assert len(set(out['roi_names'])) == 100
