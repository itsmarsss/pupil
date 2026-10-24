"""MacBook lid (hinge) angle from the built-in HID sensor, via IOKit through ctypes.

Recent MacBooks expose the hinge angle as an Apple HID sensor (vendor 0x05AC, product
0x8104, usage page 0x20 "Sensor", usage 0x8A). Feature report 1 holds the angle in
degrees as a little-endian uint16 at bytes 1–2. Not a public API; returns None on Macs
without the sensor.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging

log = logging.getLogger(__name__)

VENDOR_APPLE, PRODUCT_LID = 0x05AC, 0x8104
USAGE_PAGE_SENSOR, USAGE_LID = 0x20, 0x8A
REPORT_TYPE_FEATURE = 2

_cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
_iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))

_vp = ctypes.c_void_p
_cf.CFStringCreateWithCString.restype = _vp
_cf.CFStringCreateWithCString.argtypes = [_vp, ctypes.c_char_p, ctypes.c_uint32]
_cf.CFNumberCreate.restype = _vp
_cf.CFNumberCreate.argtypes = [_vp, ctypes.c_int, _vp]
_cf.CFDictionaryCreateMutable.restype = _vp
_cf.CFDictionaryCreateMutable.argtypes = [_vp, ctypes.c_long, _vp, _vp]
_cf.CFDictionarySetValue.argtypes = [_vp, _vp, _vp]
_cf.CFSetGetCount.restype = ctypes.c_long
_cf.CFSetGetCount.argtypes = [_vp]
_cf.CFSetGetValues.argtypes = [_vp, _vp]
_cf.CFRelease.argtypes = [_vp]
_iokit.IOHIDManagerCreate.restype = _vp
_iokit.IOHIDManagerCreate.argtypes = [_vp, ctypes.c_uint32]
_iokit.IOHIDManagerSetDeviceMatching.argtypes = [_vp, _vp]
_iokit.IOHIDManagerCopyDevices.restype = _vp
_iokit.IOHIDManagerCopyDevices.argtypes = [_vp]
_iokit.IOHIDDeviceOpen.restype = ctypes.c_int
_iokit.IOHIDDeviceOpen.argtypes = [_vp, ctypes.c_uint32]
_iokit.IOHIDDeviceGetReport.restype = ctypes.c_int
_iokit.IOHIDDeviceGetReport.argtypes = [_vp, ctypes.c_int, ctypes.c_long, _vp, ctypes.POINTER(ctypes.c_long)]

_UTF8 = 0x08000100
_SINT32 = 3
_dict_key_cb = _vp.in_dll(_cf, "kCFTypeDictionaryKeyCallBacks")
_dict_val_cb = _vp.in_dll(_cf, "kCFTypeDictionaryValueCallBacks")


def _cfstr(s: str):
    return _cf.CFStringCreateWithCString(None, s.encode(), _UTF8)


def _cfint(v: int):
    n = ctypes.c_int32(v)
    return _cf.CFNumberCreate(None, _SINT32, ctypes.byref(n))


class LidSensor:
    def __init__(self) -> None:
        self._device = None
        try:
            self._device = self._open()
        except OSError:
            log.debug("lid sensor unavailable", exc_info=True)

    @property
    def available(self) -> bool:
        return self._device is not None

    def _open(self):
        manager = _iokit.IOHIDManagerCreate(None, 0)
        match = _cf.CFDictionaryCreateMutable(None, 0, ctypes.addressof(_dict_key_cb), ctypes.addressof(_dict_val_cb))
        for key, value in (("VendorID", VENDOR_APPLE), ("ProductID", PRODUCT_LID),
                           ("PrimaryUsagePage", USAGE_PAGE_SENSOR), ("PrimaryUsage", USAGE_LID)):
            _cf.CFDictionarySetValue(match, _cfstr(key), _cfint(value))
        _iokit.IOHIDManagerSetDeviceMatching(manager, match)
        devices = _iokit.IOHIDManagerCopyDevices(manager)
        if not devices or _cf.CFSetGetCount(devices) == 0:
            return None
        buf = (_vp * _cf.CFSetGetCount(devices))()
        _cf.CFSetGetValues(devices, buf)
        for dev in buf:
            if _iokit.IOHIDDeviceOpen(dev, 0) == 0:
                self._manager = manager  # keep alive
                return dev
        return None

    def angle(self) -> float | None:
        """Hinge angle in degrees (0 = closed, ~90 = upright), or None."""
        if self._device is None:
            return None
        buf = (ctypes.c_uint8 * 8)()
        length = ctypes.c_long(8)
        if _iokit.IOHIDDeviceGetReport(self._device, REPORT_TYPE_FEATURE, 1, buf, ctypes.byref(length)) != 0:
            return None
        if length.value < 3:
            return None
        return float(buf[1] | (buf[2] << 8))
