"""Provide base class for markers."""

# Authors: Amir Omidvarnia <a.omidvarnia@fz-juelich.de>
#          Kaustubh R. Patil <k.patil@fz-juelich.de>
# License: AGPL

from typing import Dict, List
from scipy import signal as sg
import numpy as np

from ..api.decorators import register_marker
from ..utils import logger, raise_error
from .base import BaseMarker
from .parcel import ParcelAggregation

from ptpython.repl import embed # For debugging using ptpython
# print('OKKKK')
# embed(globals(), locals()) # --> In order to put a break point

@register_marker
class AmplitudeLowFrequencyFluctuationAtlas(BaseMarker):
    """Class for f/ALFF.

     Parameters
    ----------
    atlas
    agg_method
    agg_method_params
    highpass_cutoff
    lowpass_cutoff
    TR
    name

    """

    def __init__(
        self, atlas, agg_method='mean', agg_method_params=None,
        highpass_cutoff=0.01, lowpass_cutoff=0.1, TR=None, name=None
    ) -> None:
        """Initialize the class."""
        self.atlas = atlas
        self.agg_method = agg_method
        self.agg_method_params = {} if agg_method_params is None \
            else agg_method_params
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.TR = TR

        on = ["BOLD"]

        super().__init__(on=on, name=name)

    def get_meta(self, kind: str) -> Dict:
        """Get metadata.

        Parameters
        ----------
        kind : str
            The kind of pipeline step.

        Returns
        -------
        dict
            The metadata as a dictionary.

        """
        s_meta = super().get_meta()
        # same marker can be "fit"ted into different kinds, so the name
        # is created from the kind and the name of the marker
        s_meta["name"] = f"{kind}_{self.name}"
        s_meta["kind"] = kind
        return {"marker": s_meta}

    def validate_input(self, input: List[str]) -> None:
        """Validate input.

        Parameters
        ----------
        input : list of str
            The input to the pipeline step. The list must contain the
            available Junifer Data dictionary keys.

        Raises
        ------
        ValueError
            If the input does not have the required data.

        """
        if self.highpass_cutoff<0 or self.highpass_cutoff is None:
            raise_error(
                "The higher cutoff frequency must be a"
                " positive float number."
            )

        if self.lowpass_cutoff<0 or self.lowpass_cutoff is None:
            raise_error(
                "The lower cutoff frequency must be a"
                " positive float number."
            )

        if self.TR<0 or self.TR is None:
            raise_error(
                "The repetition time (TR) must be a"
                " positive float number."
            )

        if not any(x in input for x in self._valid_inputs):
            raise_error(
                "Input does not have the required data."
                f"\t Input: {input}"
                f"\t Required (any of): {self._valid_inputs}"
            )

    def get_output_kind(self, input: List[str]) -> List[str]:
        """Get output kind.

        Parameters
        ----------
        input : list of str
            The input to the marker. The list must contain the
            available Junifer Data dictionary keys.

        Returns
        -------
        list of str
            The updated list of output kinds, as storage possibilities.

        """
        outputs = ["table"]
        return outputs

    def compute(self, input: Dict) -> Dict:
        """Compute.

        Parameters
        ----------
        input : Dict[str, Dict]
            The input to the pipeline step. The list must contain the
            available Junifer Data dictionary keys.

        Returns
        -------
        A dict with 
            ROI-wise ALFF as a 1D numpy array.
            ROI-wise fALFF as a 1D numpy array.

        """
        # print('OKKKK')
        # embed(globals(), locals()) # --> In order to put a break point

        pa = ParcelAggregation(atlas=self.atlas, method=self.agg_method,
                               method_params=self.agg_method_params,
                               on="BOLD")
        # get the 2D timeseries after parcel aggregation
        ts = pa.compute(input) # N_ROI x N_T
        roi_names = ts['columns']
        ts = ts['data']

        # bandpass the data within the lowpass and highpass cutoff freqs
        Nq = np.asarray(1/(2*self.TR))
        Wn = [self.highpass_cutoff, self.lowpass_cutoff]
        Wn = np.asarray(Wn)
        Wn = Wn/Nq

        b, a = sg.butter(N=4, Wn=Wn, btype='bandpass')
        ts_filt = sg.filtfilt(b, a, ts, axis=0)

        ALFF = np.std(ts_filt, axis=0)
        PSD_tot = np.std(ts, axis=0)

        fALFF = np.divide(ALFF, PSD_tot)

        out = {}
        out["ALFF"] = ALFF
        out["fALFF"] = fALFF
        out["roi_names"] = roi_names

        return out

    # TODO: complete type annotations
    def store(self, kind: str, out: Dict, storage) -> None:
        """Store.

        Parameters
        ----------
        input
        out

        """
        logger.debug(f"Storing {kind} in {storage}")
        # storage.store_table(**out)
