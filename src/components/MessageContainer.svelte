
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: extensions ]
    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();

	// [ IMPORTS: components ]
    import Message from "./Message.svelte";
	
	// [ IMPORTS: props ]


	// [ PROPS ]
	export let messages = [];


</script>


<!-- [ TEMPLATE: MessageContainer > Chat ] -->
<div class = 'section-messages'>
<div class="scroll-area">

	{#each messages as message (message.id)}
    <div on:click={ () => { dispatch('SetMsg', message.id) } }>
        <Message 
            message={ message.message } 
            emotion={ message.emotion.assessment } 
            date={ message.date }  
            isactive={ message.is_active ? 'active' : '' }
        />
    </div>
    {/each}

</div>
</div>
