import numpy as np

__all__ = [
    'YCBCR_WEIGHTS', 'YCbCr_ranges', 'RGB_to_YCbCr', 'YCbCr_to_RGB',
    'RGB_to_YcCbcCrc', 'YcCbcCrc_to_RGB'
]


YCBCR_WEIGHTS = dict({
    'ITU-R BT.601': np.array([0.2990, 0.1140]),
    'ITU-R BT.709': np.array([0.2126, 0.0722]),
    'ITU-R BT.2020': np.array([0.2627, 0.0593]),
    'SMPTE-240M': np.array([0.2122, 0.0865])
})

WEIGHTS_YCBCR = dict(
    {
        "ITU-R BT.601": np.array([0.2990, 0.1140]),
        "ITU-R BT.709": np.array([0.2126, 0.0722]),
        "ITU-R BT.2020": np.array([0.2627, 0.0593]),
        "SMPTE-240M": np.array([0.2122, 0.0865]),
    }
)

BT2020_RGB_to_XYZ_matrix = np.asarray([[  6.36958048e-01,   1.44616904e-01,   1.68880975e-01],
       [  2.62700212e-01,   6.77998072e-01,   5.93017165e-02],
       [  4.99410657e-17,   2.80726930e-02,   1.06098506e+00]])
CAT_CAT02 = np.asarray([[ 0.7328,  0.4296, -0.1624],\
    [-0.7036,  1.6975,  0.0061],        [ 0.003 ,  0.0136,  0.9834]])



def eotf_ST2084(N,L_p=10000):
    """
    Define *SMPTE ST 2084:2014* optimised perceptual electro-optical transfer
    function (EOTF).

    This perceptual quantizer (PQ) has been modeled by Dolby Laboratories
    using *Barten (1999)* contrast sensitivity function.

    Parameters
    ----------
    N
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly

        proportional to the optical output of a display device.
    L_p
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
          Target optical output :math:`C` in :math:`cd/m^2` of the ideal
          reference display.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----
    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_ST2084(0.508078421517399)  # doctest: +ELLIPSIS
    100.0000000...
    """
    

    N = np.asarray(N).astype(np.float32)

    m_1=2610.0 / 4096.0 * (1.0 / 4.0)
    m_2=2523.0 / 4096.0 * 128.0
    c_1=3424.0 / 4096.0
    c_2=2413.0 / 4096.0 * 32.0
    c_3=2392.0 / 4096.0 * 32.0


    m_1_d = 1 / m_1
    m_2_d = 1 / m_2

    V_p = spow(N, m_2_d)
    n = np.maximum(0, V_p - c_1)
    L = spow((n / (c_2 - c_3 * V_p)), m_1_d)
    C = L_p * L

    return C.astype(np.float32)

def CV_range(bit_depth=10, is_legal=False, is_int=False):
    """
    Returns the code value :math:`CV` range for given bit depth, range legality
    and representation.

    Parameters
    ----------
    bit_depth : int, optional
        Bit depth of the code value :math:`CV` range.
    is_legal : bool, optional
        Whether the code value :math:`CV` range is legal.
    is_int : bool, optional
        Whether the code value :math:`CV` range represents integer code values.

    Returns
    -------
    ndarray
        Code value :math:`CV` range.

    Examples
    --------
    >>> CV_range(8, True, True)
    array([ 16, 235])
    >>> CV_range(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...])
    >>> CV_range(10, False, False)
    array([ 0.,  1.])
    """

    if is_legal:
        ranges = np.array([16, 235])
        ranges *= 2 ** (bit_depth - 8)
    else:
        ranges = np.array([0, 2 ** bit_depth - 1])

    if not is_int:
        ranges = ranges.astype(np.float32) / (2 ** bit_depth - 1)

    return ranges


def YCbCr_ranges(bits, is_legal, is_int):
    """"
    Returns the *Y'CbCr* colour encoding ranges array for given bit depth,
    range legality and representation.

    Parameters
    ----------
    bits : int
        Bit depth of the *Y'CbCr* colour encoding ranges array.
    is_legal : bool
        Whether the *Y'CbCr* colour encoding ranges array is legal.
    is_int : bool
        Whether the *Y'CbCr* colour encoding ranges array represents integer
        code values.

    Returns
    -------
    ndarray
        *Y'CbCr* colour encoding ranges array.

    Examples
    --------
    >>> YCbCr_ranges(8, True, True)
    array([ 16, 235,  16, 240])
    >>> YCbCr_ranges(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...,  0.0627451...,  0.9411764...])
    >>> YCbCr_ranges(10, False, False)
    array([ 0. ,  1. , -0.5,  0.5])
    """

    if is_legal:
        ranges = np.array([16, 235, 16, 240])
        ranges *= 2 ** (bits - 8)
    else:
        ranges = np.array([0, 2 ** bits - 1, 0, 2 ** bits - 1])

    if not is_int:
        ranges = ranges.astype(np.float32) / (2 ** bits - 1)

    if is_int and not is_legal:
        ranges[3] = 2 ** bits

    if not is_int and not is_legal:
        ranges[2] = -0.5
        ranges[3] = 0.5

    return ranges.astype(np.float32)
def eotf_PQ_BT2100(E_p):
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* electro-optical
    transfer function (EOTF).

    The EOTF maps the non-linear *PQ* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_PQ_BT2100(0.724769816665726)  # doctest: +ELLIPSIS
    779.9883608...
    """

    return eotf_ST2084(E_p, 10000)


def RGB_to_YCbCr(
    RGB,
    K= WEIGHTS_YCBCR["ITU-R BT.709"],
    in_bits= 10,
    in_legal= False,
    in_int= False,
    out_bits= 8,
    out_legal= True,
    out_int= False,
    **kwargs):


    Kr, Kb = K
    RGB_min, RGB_max = CV_range(in_bits, in_legal, in_int)
    Y_min, Y_max, C_min, C_max = YCbCr_ranges(out_bits, out_legal, out_int)

    RGB_float = RGB.astype(np.float32) - RGB_min
    RGB_float *= 1 / (RGB_max - RGB_min)
    R, G, B = RGB_float[:,:,0],RGB_float[:,:,1],RGB_float[:,:,2]

    Y = Kr * R + (1 - Kr - Kb) * G + Kb * B
    Cb = 0.5 * (B - Y) / (1 - Kb)
    Cr = 0.5 * (R - Y) / (1 - Kr)
    Y *= Y_max - Y_min
    Y += Y_min
    Cb *= C_max - C_min
    Cr *= C_max - C_min
    Cb += (C_max + C_min) / 2
    Cr += (C_max + C_min) / 2

    YCbCr = np.stack([Y, Cb, Cr],axis=2)

    return YCbCr


def YCbCr_to_RGB(YCbCr,
                 K=YCBCR_WEIGHTS['ITU-R BT.709'],
                 in_bits=8,
                 in_legal=True,
                 in_int=False,
                 out_bits=10,
                 out_legal=False,
                 out_int=False,
                 **kwargs):
    """
    Converts an array of *Y'CbCr* colour encoding values to the corresponding
    *R'G'B'* values array.

    Parameters
    ----------
    YCbCr : array_like
        Input *Y'CbCr* colour encoding array of integer or float values.
    K : array_like, optional
        Luma weighting coefficients of red and blue. See
        :attr:`colour.YCBCR_WEIGHTS` for presets. Default is
        *(0.2126, 0.0722)*, the weightings for *ITU-R BT.709*.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Default is *8*.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is *True*.
    in_int : bool, optional
        Whether to treat the input values as ``in_bits`` integer code values.
        Default is *False*.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Ignored if ``out_legal`` and
        ``out_int`` are both *False*. Default is *10*.
    out_legal : bool, optional
        Whether to return legal range values. Default is *False*.
    out_int : bool, optional
        Whether to return values as ``out_bits`` integer code values. Default
        is *False*.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        *in_range = (Y_min, Y_max, C_min, C_max)*. If ``in_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed using
        :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        *out_range = (RGB_min, RGB_max)*. If ``out_range`` is undefined,
        *RGB_min* and *RGB_max* will be computed using :func:`colour.CV_range`
        definition.

    Returns
    -------
    ndarray
        *R'G'B'* array of integer or float values.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YCbCr``      | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`colour.YCbCr_to_RGB`
    definition is only applicable to the non-constant luminance implementation.
    :func:`colour.YcCbcCrc_to_RGB` definition should be used for the constant
    luminance case as per :cite:`InternationalTelecommunicationUnion2015h`.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011e`,
    :cite:`InternationalTelecommunicationUnion2015i`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,
    :cite:`Wikipedia2004d`

    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_RGB(YCbCr, in_bits=10, in_legal=True, in_int=True)
    array([ 0.5,  0.5,  0.5])
    """

    YCbCr = YCbCr.astype(np.float32)
    Y, Cb, Cr = YCbCr[:,:,0], YCbCr[:,:,1], YCbCr[:,:,2],
    Kr, Kb = K
    Y_min, Y_max, C_min, C_max = YCbCr_ranges(in_bits, in_legal, in_int)
    RGB_min, RGB_max = CV_range(out_bits, out_legal, out_int)

    Y -= Y_min
    Cb -= (C_max + C_min) / 2
    Cr -= (C_max + C_min) / 2
    Y *= 1 / (Y_max - Y_min)
    Cb *= 1 / (C_max - C_min)
    Cr *= 1 / (C_max - C_min)
    R = Y + (2 - 2 * Kr) * Cr
    B = Y + (2 - 2 * Kb) * Cb
    G = (Y - Kr * R - Kb * B) / (1 - Kr - Kb)

    RGB = np.dstack([R, G, B])
    RGB *= RGB_max - RGB_min
    RGB += RGB_min
    RGB = np.round(RGB).astype(np.uint16) if out_int else RGB

    return RGB

def matrix_chromatic_adaptation_VonKries(
    XYZ_w,\
    XYZ_wr,\
    transform):
    """
    Compute the *chromatic adaptation* matrix from test viewing conditions
    to reference viewing conditions.

    Parameters
    ----------
    XYZ_w
        Test viewing conditions *CIE XYZ* tristimulus values of whitepoint.
    XYZ_wr
        Reference viewing conditions *CIE XYZ* tristimulus values of
        whitepoint.
    transform
        Chromatic adaptation transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatic adaptation matrix :math:`M_{cat}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_w``  | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_wr`` | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2013t`

    Examples
    --------
    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr)
    ... # doctest: +ELLIPSIS
    array([[ 1.0425738...,  0.0308910..., -0.0528125...],
           [ 0.0221934...,  1.0018566..., -0.0210737...],
           [-0.0011648..., -0.0034205...,  0.7617890...]])

    Using Bradford method:

    >>> XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
    >>> XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
    >>> method = 'Bradford'
    >>> matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr, method)
    ... # doctest: +ELLIPSIS
    array([[ 1.0479297...,  0.0229468..., -0.0501922...],
           [ 0.0296278...,  0.9904344..., -0.0170738...],
           [-0.0092430...,  0.0150551...,  0.7518742...]])
    """



    if(transform=='CAT02'):
        M = CAT_CAT02

    RGB_w = np.einsum("...i,...ij->...j", XYZ_w, np.transpose(M))
    RGB_wr = np.einsum("...i,...ij->...j", XYZ_wr, np.transpose(M))

    D = RGB_wr / RGB_w

    D = np.expand_dims(D, -2)

    D = np.eye(D.shape[-1]) * D

    M_CAT = np.einsum("...ij,...jk->...ik",np.linalg.inv(M), D )
    M_CAT = np.einsum("...ij,...jk->...ik",M_CAT, M )

    return M_CAT

def xyY_to_XYZ(xyY):
    """
    Convert from *CIE xyY* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Lindbloom2009d`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> xyY = np.array([0.54369557, 0.32107944, 0.12197225])
    >>> xyY_to_XYZ(xyY)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    x, y, Y = np.array([xyY[..., i] for i in range(xyY.shape[-1])])  

    XYZ = np.where(
        (y == 0)[..., np.newaxis],
        np.stack([y, y, y],-1),
        np.stack([x * Y / y, Y, (1 - x - y) * Y / y],-1),
    )

    return XYZ

def xy_to_xyY(xy,Y=1):
    """
    Convert from *CIE xy* chromaticity coordinates to *CIE xyY* colourspace by
    extending the array last dimension with given :math:`Y` *luminance*.

    ``xy`` argument with last dimension being equal to 3 will be assumed to be
    a *CIE xyY* colourspace array argument and will be returned directly by the
    definition.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array.
    Y
        Optional :math:`Y` *luminance* value used to construct the *CIE xyY*
        colourspace array, the default :math:`Y` *luminance* value is 1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xyY* colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xy``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   This definition is a convenient object provided to implement support of
        illuminant argument *luminance* value in various :mod:`colour.models`
        package objects such as :func:`colour.Lab_to_XYZ` or
        :func:`colour.Luv_to_XYZ`.

    References
    ----------
    :cite:`Wikipedia2005`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_xyY(xy)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  1.        ])
    >>> xy = np.array([0.54369557, 0.32107944, 1.00000000])
    >>> xy_to_xyY(xy)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  1.        ])
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_xyY(xy, 100)  # doctest: +ELLIPSIS
    array([   0.5436955...,    0.3210794...,  100.        ])
    """

    xy = np.asarray(xy).astype(np.float32)
    Y = np.asarray(Y).astype(np.float32)

    # Assuming ``xy`` is actually a *CIE xyY* colourspace array argument and
    # returning it directly.
    if xy.shape[-1] == 3:
        return xy

    x, y = xy[...,0],xy[...,1]

    xyY = np.stack([x, y, np.full(x.shape, Y)],-1)

    return xyY


def RGB_to_XYZ(RGB,illuminant_RGB,illuminant_XYZ,matrix_RGB_to_XYZ,chromatic_adaptation_transform,cctf_decoding):
    """
    Convert given *RGB* colourspace array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    illuminant_RGB
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the input *RGB* colourspace array.
    illuminant_XYZ
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array of the
        *illuminant* for the output *CIE XYZ* tristimulus values.
    matrix_RGB_to_XYZ
        Matrix converting the *RGB* colourspace array to *CIE XYZ* tristimulus
        values, i.e. the *Normalised Primary Matrix* (NPM).
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    cctf_decoding
        Decoding colour component transfer function (Decoding CCTF) or
        electro-optical transfer function (EOTF).

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +--------------------+-----------------------+---------------+
    | **Domain**         | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``RGB``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_XYZ`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+
    | ``illuminant_RGB`` | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    +--------------------+-----------------------+---------------+
    | **Range**          | **Scale - Reference** | **Scale - 1** |
    +====================+=======================+===============+
    | ``XYZ``            | [0, 1]                | [0, 1]        |
    +--------------------+-----------------------+---------------+

    Examples
    --------
    >>> RGB = np.array([0.45595571, 0.03039702, 0.04087245])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> matrix_RGB_to_XYZ = np.array(
    ...     [[0.41240000, 0.35760000, 0.18050000],
    ...      [0.21260000, 0.71520000, 0.07220000],
    ...      [0.01930000, 0.11920000, 0.95050000]]
    ... )
    >>> RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, matrix_RGB_to_XYZ,
    ...            chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.2163881...,  0.1257    ,  0.0384749...])
    """


    if cctf_decoding is not None:
        RGB = np.stack((eotf_PQ_BT2100(RGB[:,:,0]),eotf_PQ_BT2100(RGB[:,:,1]),eotf_PQ_BT2100(RGB[:,:,2])),axis=2)

    XYZ = np.einsum("...ij,...j->...i", matrix_RGB_to_XYZ, RGB)

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
            xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
            transform=chromatic_adaptation_transform,
        )

        XYZ = np.einsum("...ij,...j->...i", M_CAT, XYZ)

    return XYZ
    
def reaction_rate_MichaelisMenten_Michaelis1913(S,V_max,K_m):

    v = (V_max * S) / (K_m + S)

    return v



def lightness_Fairchild2010(Y, epsilon = 1.836):
    """
    Compute *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Wyble (2010)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.
    epsilon
        :math:`\\epsilon` exponent.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Lightness* :math:`L_{hdr}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2010`

    Examples
    --------
    >>> lightness_Fairchild2010(12.19722535 / 100)  # doctest: +ELLIPSIS
    31.9963902...
    """


    maximum_perception = 100

    L_hdr = (
        reaction_rate_MichaelisMenten_Michaelis1913(
            spow(Y, epsilon), maximum_perception, spow(0.184, epsilon)
        )
        + 0.02
    )

    return L_hdr

def spow(a,p):

    a_p = np.sign(a) * np.abs(a) ** p

    a_p = np.nan_to_num(a_p)
    return a_p


def lightness_Fairchild2011(Y,epsilon= 0.474,method= "hdr-CIELAB"):
    """
    Compute *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Chen (2011)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    Y
        *Luminance* :math:`Y`.
    epsilon
        :math:`\\epsilon` exponent.
    method
        *Lightness* :math:`L_{hdr}` computation method.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *Lightness* :math:`L_{hdr}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2011`

    Examples
    --------
    >>> lightness_Fairchild2011(12.19722535 / 100)  # doctest: +ELLIPSIS
    51.8529584...
    >>> lightness_Fairchild2011(12.19722535 / 100, method='hdr-IPT')
    ... # doctest: +ELLIPSIS
    51.6431084...
    """


    if method == "hdr-cielab":
        maximum_perception = 247
    else:
        maximum_perception = 246

    L_hdr = (
        reaction_rate_MichaelisMenten_Michaelis1913(
            spow(Y, epsilon), maximum_perception, spow(2, epsilon)
        )
        + 0.02
    )

    return L_hdr


def exponent_hdr_CIELab(Y_s,Y_abs,method):
    """
    Compute *hdr-CIELAB* colourspace *Lightness* :math:`\\epsilon` exponent
    using *Fairchild and Wyble (2010)* or *Fairchild and Chen (2011)* method.

    Parameters
    ----------
    Y_s
        Relative luminance :math:`Y_s` of the surround.
    Y_abs
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *hdr-CIELAB* colourspace *Lightness* :math:`\\epsilon` exponent.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y_s``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> exponent_hdr_CIELab(0.2, 100)  # doctest: +ELLIPSIS
    0.4738510...
    >>> exponent_hdr_CIELab(0.2, 100, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    1.8360198...
    """


    if method == "fairchild 2010":
        epsilon = 1.50
    else:
        epsilon = 0.58

    sf = 1.25 - 0.25 * (Y_s / 0.184)
    lf = np.log(318) / np.log(Y_abs)
    if method == "fairchild 2010":
        epsilon *= sf * lf
    else:
        epsilon /= sf * lf

    return epsilon


def XYZ_to_hdr_CIELab(XYZ,illuminant,Y_s,Y_abs,method):

    X, Y, Z = XYZ[...,0],XYZ[...,1],XYZ[...,2]

    XYZ_n = xyY_to_XYZ(xy_to_xyY(illuminant))
    X_n, Y_n, Z_n = XYZ_n[...,0],XYZ_n[...,1],XYZ_n[...,2]

    if method == "fairchild 2010":
        lightness_callable = lightness_Fairchild2010
    else:
        lightness_callable = lightness_Fairchild2011

    e = exponent_hdr_CIELab(Y_s, Y_abs, method)

    # Domain and range scaling has already be handled.
    L_hdr = lightness_callable(Y / Y_n, e)
    a_hdr = 5 * (lightness_callable(X / X_n, e) - L_hdr)
    b_hdr = 2 * (L_hdr - lightness_callable(Z / Z_n, e))

    Lab_hdr = np.stack([L_hdr, a_hdr, b_hdr],-1)

    return Lab_hdr