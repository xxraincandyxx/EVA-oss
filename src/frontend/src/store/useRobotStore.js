import { create } from 'zustand';

const useRobotStore = create((set, get) => ({
  // State
  isOnline: false,
  status: {
    position: [0, 0, 0],
    orientation: [0, 0, 0],
    thetas: [0, 0, 0, 0, 0, 0],
  },
  schedule: [],
  isTracking: false,
  isRepeating: false,
  simulationStates: [],
  interactivePreview: null,
  interactiveError: '',
  socket: null,
  agentMessages: [],
  isAgentTyping: false,
  activePage: 'analytics',
  llmBackend: 'deepseek',
  localLlmUrl: '',

  // Actions
  setSocket: (socket) => set({ socket }),

  setActivePage: (page) => set({ activePage: page }),

  initializeListeners: (socket) => {
    const listeners = {
      connect: () => set({ isOnline: true }),
      disconnect: () => set({ isOnline: false }),
      update_system_online_status: (data) => set({ isOnline: data.online }),
      status_update: (data) => set({ status: data }),
      caches_update: (data) =>
        set({ schedule: data.scheduled_cartesians ?? [] }),
      tracking_status_update: (data) =>
        set({ isTracking: data.is_tracking }),
      repeat_status_update: (data) =>
        set({ isRepeating: data.is_repeating }),
      update_arm_sim_data: (data) =>
        set({ simulationStates: data.states_list ?? [] }),
      interactive_preview_result: (data) =>
        set({ interactivePreview: data, interactiveError: '' }),
      interactive_preview_error: (data) =>
        set({ interactiveError: data.message ?? 'Target is unreachable.' }),
      agent_response: (data) =>
        set((state) => ({
          isAgentTyping: false,
          agentMessages: [
            ...state.agentMessages,
            { sender: 'bot', text: data.response },
          ],
        })),
      llm_backend_changed: (data) =>
        set({
          llmBackend: data.backend,
          localLlmUrl: data.local_url || '',
        }),
    };

    for (const [event, listener] of Object.entries(listeners)) {
      socket.on(event, listener);
    }

    return () => {
      for (const [event, listener] of Object.entries(listeners)) {
        socket.off(event, listener);
      }
      set((state) => ({
        isOnline: false,
        socket: state.socket === socket ? null : state.socket,
      }));
    };
  },

  sendAgentMessage: (messageText) => {
    const { socket, agentMessages } = get();
    if (socket) {
      set({
        isAgentTyping: true,
        agentMessages: [
          ...agentMessages,
          { sender: 'user', text: messageText },
        ],
      });
      socket.emit('send_agent_message', { message: messageText });
    }
  },

  emit: (event, args) => {
    const { socket } = get();
    if (socket) {
      socket.emit(event, args);
    } else {
      console.error('Socket not connected. Cannot emit event:', event);
    }
  },

  clearInteractivePreview: () =>
    set({ interactivePreview: null, interactiveError: '' }),

  setLlmBackend: (backend, localUrl) => {
    const { socket } = get();
    if (socket) {
      socket.emit('set_llm_backend', {
        backend,
        local_url: localUrl,
      });
      set({ llmBackend: backend, localLlmUrl: localUrl });
    }
  },
}));

export default useRobotStore;
