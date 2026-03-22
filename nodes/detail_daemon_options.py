from .utils.detail_daemon import DD_DEFAULTS


class DetailDaemonOptions:
    """Optional configuration node for Detail Daemon advanced parameters.
    Connect to PersonDetailer's dd_options input to override defaults."""

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = ("DD_OPTIONS",)
    RETURN_NAMES = ("dd_options",)
    DESCRIPTION = (
        "Fine-tune Detail Daemon sigma manipulation parameters.\n\n"
        "Connect to Person Detailer's dd_options input to override the built-in defaults.\n"
        "Controls the sigma modulation curve shape, range, bias, and fade."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dd_start": ("FLOAT", {"default": DD_DEFAULTS["dd_start"], "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Start of the sigma range to modulate (0.0-1.0)"}),
                "dd_end": ("FLOAT", {"default": DD_DEFAULTS["dd_end"], "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "End of the sigma range to modulate (0.0-1.0)"}),
                "dd_bias": ("FLOAT", {"default": DD_DEFAULTS["dd_bias"], "min": 0.01, "max": 5.0, "step": 0.01,
                                       "tooltip": "Bias shifts the modulation curve"}),
                "dd_exponent": ("FLOAT", {"default": DD_DEFAULTS["dd_exponent"], "min": 0.01, "max": 5.0, "step": 0.01,
                                           "tooltip": "Exponent controls the curve shape"}),
                "dd_start_offset": ("FLOAT", {"default": DD_DEFAULTS["dd_start_offset"], "min": -0.5, "max": 0.5, "step": 0.01,
                                               "tooltip": "Offset applied to start boundary"}),
                "dd_end_offset": ("FLOAT", {"default": DD_DEFAULTS["dd_end_offset"], "min": -0.5, "max": 0.5, "step": 0.01,
                                             "tooltip": "Offset applied to end boundary"}),
                "dd_fade": ("FLOAT", {"default": DD_DEFAULTS["dd_fade"], "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Fade in/out at range boundaries"}),
            }
        }

    def execute(self, dd_start, dd_end, dd_bias, dd_exponent, dd_start_offset, dd_end_offset, dd_fade):
        return ({
            "dd_start": dd_start,
            "dd_end": dd_end,
            "dd_bias": dd_bias,
            "dd_exponent": dd_exponent,
            "dd_start_offset": dd_start_offset,
            "dd_end_offset": dd_end_offset,
            "dd_fade": dd_fade,
        },)
