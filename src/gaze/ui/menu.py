"""The ◉ menu-bar item: a status line, then sections of actions, toggles and submenus.

Items are declared as data (`Action`, `Toggle`, `Choice`, `Header`, `SEPARATOR`) and the whole menu is rebuilt
each time it opens, so checkmarks, titles and device lists are always current.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import objc
from AppKit import NSMenu, NSMenuItem, NSObject, NSStatusBar, NSVariableStatusItemLength


@dataclass
class Action:
    title: str | Callable[[], str]
    run: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True


@dataclass
class Toggle:
    title: str
    is_on: Callable[[], bool]
    run: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True


@dataclass
class Choice:
    """A submenu of options, one of them checked. `options()` → [(title, selected, run)]."""
    title: str
    options: Callable[[], list[tuple[str, bool, Callable[[], None]]]]


@dataclass
class Header:
    title: str


SEPARATOR = None
Item = Union[Action, Toggle, Choice, Header, None]


class _Target(NSObject):
    """Bridges Objective-C menu callbacks to Python: actions by tag, and rebuilding on open."""

    def initWithOwner_(self, owner):
        self = objc.super(_Target, self).init()
        if self is not None:
            self.owner = owner
        return self

    def run_(self, sender):
        self.owner._run(sender.tag())

    def menuNeedsUpdate_(self, menu):
        self.owner._rebuild(menu)


class MenuBar:
    def __init__(self, items: Callable[[], list[Item]], status: Callable[[], str]):
        """items() is called each time the menu opens, so entries can come and go."""
        self.items, self.status = items, status
        self._actions: list[Callable[[], None]] = []
        self._target = _Target.alloc().initWithOwner_(self)
        self._menu = NSMenu.alloc().init()
        self._menu.setDelegate_(self._target)
        self._menu.setAutoenablesItems_(False)
        self._rebuild(self._menu)
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._status_item.button().setTitle_("◉")
        self._status_item.setMenu_(self._menu)

    def _run(self, tag: int) -> None:
        if 0 <= tag < len(self._actions):
            self._actions[tag]()

    def _add(self, menu, title: str, run: Callable[[], None] | None, checked: bool | None = None,
             enabled: bool = True) -> NSMenuItem:
        item = menu.addItemWithTitle_action_keyEquivalent_(title, "run:" if run else None, "")
        if run:
            item.setTarget_(self._target)
            item.setTag_(len(self._actions))
            self._actions.append(run)
        if checked is not None:
            item.setState_(1 if checked else 0)
        item.setEnabled_(enabled and run is not None)
        return item

    def _rebuild(self, menu) -> None:
        menu.removeAllItems()
        self._actions = []
        self._add(menu, self.status(), None)
        menu.addItem_(NSMenuItem.separatorItem())
        for item in self.items():
            if item is SEPARATOR:
                menu.addItem_(NSMenuItem.separatorItem())
            elif isinstance(item, Header):
                if hasattr(NSMenuItem, "sectionHeaderWithTitle_"):  # macOS 14+
                    menu.addItem_(NSMenuItem.sectionHeaderWithTitle_(item.title))
                else:
                    self._add(menu, item.title, None)
            elif isinstance(item, Action):
                title = item.title() if callable(item.title) else item.title
                self._add(menu, title, item.run, enabled=item.enabled())
            elif isinstance(item, Toggle):
                self._add(menu, item.title, item.run, checked=item.is_on(), enabled=item.enabled())
            elif isinstance(item, Choice):
                parent = menu.addItemWithTitle_action_keyEquivalent_(item.title, None, "")
                sub = NSMenu.alloc().init()
                sub.setAutoenablesItems_(False)
                for title, selected, run in item.options():
                    self._add(sub, title, run, checked=selected)
                parent.setSubmenu_(sub)
