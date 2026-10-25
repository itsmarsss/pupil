"""Where the Mac is: macOS Location Services, turned into an address (postal code, city, country) by Apple's geocoder.

For requests the screen can't answer ("what's my postal code", "how far is the airport"). Personal data: the agent's
location tool always waits for the user's "confirm" first (agent.ASK_FIRST), and macOS asks once whether this app
(your terminal) may use your location at all. Nothing here goes anywhere but Apple's own services.
"""
from __future__ import annotations

import time

TIMEOUT_S = 12.0


def _spin(until, timeout_s):
    """Run this thread's run loop until `until()` or the timeout: CoreLocation answers through it."""
    from Foundation import NSDate, NSRunLoop

    end = time.time() + timeout_s
    while not until() and time.time() < end:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))


def current(timeout_s: float = TIMEOUT_S) -> tuple[str, bool]:
    """(where the Mac is, in words; is_error)."""
    try:
        import CoreLocation
    except ImportError:
        return ("Location isn't available here (pyobjc-framework-CoreLocation missing). Open the Maps app and use "
                "its current location instead.", True)

    if not CoreLocation.CLLocationManager.locationServicesEnabled():
        return "Location Services are off (System Settings → Privacy & Security → Location Services).", True
    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = _delegate_class().alloc().init()
    manager.setDelegate_(delegate)
    status = manager.authorizationStatus()
    if status == CoreLocation.kCLAuthorizationStatusNotDetermined:
        manager.requestWhenInUseAuthorization()  # macOS asks the user, once
        _spin(lambda: manager.authorizationStatus() != CoreLocation.kCLAuthorizationStatusNotDetermined, 30)
    if manager.authorizationStatus() in (CoreLocation.kCLAuthorizationStatusDenied,
                                         CoreLocation.kCLAuthorizationStatusRestricted):
        return ("Location access is off for this app (System Settings → Privacy & Security → Location Services). "
                "Open the Maps app and use its current location instead.", True)
    manager.requestLocation()
    _spin(lambda: delegate.location is not None or delegate.error is not None, timeout_s)
    if delegate.location is None:
        return f"Couldn't get the location ({delegate.error or 'timed out'}). Try the Maps app.", True
    return _address(delegate.location, timeout_s)


_DELEGATE = None


def _delegate_class():
    """The CoreLocation delegate class, made once (Objective-C classes are registered by name)."""
    global _DELEGATE
    if _DELEGATE is None:
        import objc
        from Foundation import NSObject

        class PupilLocationDelegate(NSObject):
            def init(self):
                self = objc.super(PupilLocationDelegate, self).init()
                if self is not None:
                    self.location, self.error = None, None
                return self

            def locationManager_didUpdateLocations_(self, manager, locations):
                self.location = locations[-1]

            def locationManager_didFailWithError_(self, manager, error):
                self.error = str(error.localizedDescription())

            def locationManagerDidChangeAuthorization_(self, manager):
                pass

        _DELEGATE = PupilLocationDelegate
    return _DELEGATE


def _address(location, timeout_s):
    import CoreLocation

    found = {}
    CoreLocation.CLGeocoder.alloc().init().reverseGeocodeLocation_completionHandler_(
        location, lambda marks, error: found.update(marks=marks, error=error))
    _spin(lambda: bool(found), timeout_s)
    coord = location.coordinate()
    where = f"{coord.latitude:.5f}, {coord.longitude:.5f} (within ~{round(location.horizontalAccuracy())} m)"
    marks = found.get("marks")
    if not marks:
        return f"Coordinates {where}; no address for them ({found.get('error') or 'timed out'}).", False
    m = marks[0]
    parts = {"street": " ".join(p for p in (m.subThoroughfare(), m.thoroughfare()) if p),
             "city": m.locality(), "region": m.administrativeArea(), "postal_code": m.postalCode(),
             "country": m.country()}
    return "; ".join(f"{k}: {v}" for k, v in parts.items() if v) + f"; coordinates: {where}", False
