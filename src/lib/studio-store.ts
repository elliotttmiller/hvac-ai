/**
 * Studio State Management
 * 
 * Centralized Zustand store for the unified Studio experience.
 * Manages panel states, selection, hover, and visibility toggles.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface StudioPanel {
  isCollapsed: boolean;
  size: number; // percentage or pixels
}

export interface ComponentVisibility {
  [classLabel: string]: boolean;
}

export interface StudioState {
  // Panel States
  navigatorPanel: StudioPanel;
  inspectorPanel: StudioPanel;
  
  // Selection & Interaction
  selectedComponentId: string | null;
  hoveredComponentId: string | null;
  
  // Component Visibility (by class label)
  componentVisibility: ComponentVisibility;
  
  // Actions
  setNavigatorCollapsed: (collapsed: boolean) => void;
  setInspectorCollapsed: (collapsed: boolean) => void;
  setNavigatorSize: (size: number) => void;
  setInspectorSize: (size: number) => void;
  
  setSelectedComponent: (id: string | null) => void;
  setHoveredComponent: (id: string | null) => void;
  
  toggleComponentVisibility: (classLabel: string) => void;
  setComponentVisibility: (classLabel: string, visible: boolean) => void;
  setAllComponentsVisible: (visible: boolean) => void;
  
  // Reset to defaults
  resetPanels: () => void;
}

const DEFAULT_NAVIGATOR_SIZE = 20; // 20% width
const DEFAULT_INSPECTOR_SIZE = 25; // 25% width

export const useStudioStore = create<StudioState>()(
  persist(
    (set, get) => ({
      // Initial State
      navigatorPanel: {
        isCollapsed: false,
        size: DEFAULT_NAVIGATOR_SIZE,
      },
      inspectorPanel: {
        isCollapsed: false,
        size: DEFAULT_INSPECTOR_SIZE,
      },
      
      selectedComponentId: null,
      hoveredComponentId: null,
      componentVisibility: {},
      
      // Panel Actions
      setNavigatorCollapsed: (collapsed) =>
        set((state) => ({
          navigatorPanel: { ...state.navigatorPanel, isCollapsed: collapsed },
        })),
      
      setInspectorCollapsed: (collapsed) =>
        set((state) => ({
          inspectorPanel: { ...state.inspectorPanel, isCollapsed: collapsed },
        })),
      
      setNavigatorSize: (size) =>
        set((state) => ({
          navigatorPanel: { ...state.navigatorPanel, size },
        })),
      
      setInspectorSize: (size) =>
        set((state) => ({
          inspectorPanel: { ...state.inspectorPanel, size },
        })),
      
      // Selection Actions
      setSelectedComponent: (id) => set({ selectedComponentId: id }),
      setHoveredComponent: (id) => set({ hoveredComponentId: id }),
      
      // Visibility Actions
      toggleComponentVisibility: (classLabel) =>
        set((state) => ({
          componentVisibility: {
            ...state.componentVisibility,
            [classLabel]: !(state.componentVisibility[classLabel] ?? true),
          },
        })),
      
      setComponentVisibility: (classLabel, visible) =>
        set((state) => ({
          componentVisibility: {
            ...state.componentVisibility,
            [classLabel]: visible,
          },
        })),
      
      setAllComponentsVisible: (visible) => {
        const { componentVisibility } = get();
        const updated = Object.keys(componentVisibility).reduce(
          (acc, key) => {
            acc[key] = visible;
            return acc;
          },
          {} as ComponentVisibility
        );
        set({ componentVisibility: updated });
      },
      
      // Reset
      resetPanels: () =>
        set({
          navigatorPanel: {
            isCollapsed: false,
            size: DEFAULT_NAVIGATOR_SIZE,
          },
          inspectorPanel: {
            isCollapsed: false,
            size: DEFAULT_INSPECTOR_SIZE,
          },
          selectedComponentId: null,
          hoveredComponentId: null,
        }),
    }),
    {
      name: 'hvac-studio-state',
      partialize: (state) => ({
        navigatorPanel: state.navigatorPanel,
        inspectorPanel: state.inspectorPanel,
        componentVisibility: state.componentVisibility,
      }),
    }
  )
);
