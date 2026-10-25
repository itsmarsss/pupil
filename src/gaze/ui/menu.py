"""The ◉ menu-bar item: a disabled status line, then actions."""
from __future__ import annotations

from typing import Callable

import objc
from AppKit import NSMenu, NSMenuItem, NSObject, NSStatusBar, NSVariableStatusItemLength


class _MenuTarget(NSObject):
    """Bridges menu actions (Objective-C selectors) to Python callables."""

    def initWithActions_(self, actions):
        self = objc.super(_MenuTarget, self).init()
        if self is not None:
            self.actions = actions
        return self

    def run_(self, sender):
        self.actions[sender.tag()]()


class MenuBar:
    """`items`: (key, title, action) in order; a key of "-" adds a separator."""

    def __init__(self, items: list[tuple[str, str, Callable[[], None] | None]]):
        actions = [action for key, _, action in items if key != "-"]
        self._target = _MenuTarget.alloc().initWithActions_(actions)
        menu = NSMenu.alloc().init()
        self._status = menu.addItemWithTitle_action_keyEquivalent_("Starting…", None, "")
        self._status.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())
        self._items: dict[str, NSMenuItem] = {}
        tag = 0
        for key, title, _ in items:
            if key == "-":
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = menu.addItemWithTitle_action_keyEquivalent_(title, "run:", "")
            item.setTarget_(self._target)
            item.setTag_(tag)
            self._items[key] = item
            tag += 1
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._status_item.button().setTitle_("◉")
        self._status_item.setMenu_(menu)

    def set_status(self, text: str) -> None:
        self._status.setTitle_(text)

    def set_checked(self, key: str, on: bool) -> None:
        self._items[key].setState_(1 if on else 0)

    def set_title(self, key: str, title: str) -> None:
        self._items[key].setTitle_(title)
